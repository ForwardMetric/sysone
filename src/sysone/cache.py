"""The on-disk feature cache for a frozen backbone (`cache_dir`).

Vectors are grouped into **sets**, one folder per combination of everything that determines them
(base model and revision, kept layers, layout, precision, prompt format, token limits). Within a
set, each question is looked up by a hash of its encoded tokens, so growing the data only
computes the new questions. Each write adds one shard file to the set.

The whole folder is capped at `cache_max_gb`: the least recently used sets are deleted first, and
a set that would exceed the cap on its own is never written. Nothing is written if it would
leave less than 10% of the disk free.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from array import array
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import torch

from ._util import format_bytes
from .features import CACHE_DTYPE, Features
from .layout import Encoded

FORMAT = 1
SET_PREFIX = "set-"
META_FILE = "set.json"
#: Writing is skipped when it would leave less than this fraction of the disk free.
MIN_FREE_FRACTION = 0.10


def question_key(encoded: Encoded) -> bytes:
    """A 16-byte hash of everything about an encoded question that its vectors depend on."""
    h = hashlib.blake2b(digest_size=16)
    for values in (encoded.input_ids, encoded.position_ids, encoded.option_ids, encoded.read_positions,
                   [encoded.summary_position]):
        h.update(array("q", values).tobytes())
        h.update(b"|")
    return h.digest()


def _set_name(parts: dict) -> str:
    text = json.dumps({"format": FORMAT, **parts}, sort_keys=True, default=str)
    return SET_PREFIX + hashlib.sha256(text.encode()).hexdigest()[:16]


def _folder_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _read_meta(path: Path) -> dict | None:
    try:
        with open(path / META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _sets(root: Path) -> list[tuple[Path, dict]]:
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.glob(SET_PREFIX + "*")):
        meta = _read_meta(path) if path.is_dir() else None
        if meta is not None:
            found.append((path, meta))
    return found


class FeatureCache:
    """One set of cached vectors in `root`, identified by `parts`."""

    def __init__(self, root, parts: dict, max_bytes: float, log: Callable[[str], None]):
        self.root = Path(root).expanduser()
        self.parts = parts
        self.path = self.root / _set_name(parts)
        self.max_bytes = max_bytes
        self.log = log

    def _touch(self, meta: dict) -> None:
        meta["last_used"] = time.time()
        _write_json(self.path / META_FILE, meta)

    def lookup(self, keys: Sequence[bytes]) -> dict[bytes, Features]:
        """The cached vectors for whichever of `keys` are in this set."""
        from safetensors.torch import load_file

        meta = _read_meta(self.path)
        if meta is None:
            return {}
        wanted, found = set(keys), {}
        for shard in sorted(self.path.glob("shard-*.safetensors")):
            data = load_file(str(shard))
            offsets = data["offsets"].tolist()
            for i, key in enumerate(bytes(row) for row in data["keys"].numpy()):
                if key in wanted and key not in found:
                    found[key] = Features(data["options"][offsets[i]: offsets[i + 1]].clone(),
                                          data["summary"][i].clone())
        self._touch(meta)
        return found

    def add(self, keys: Sequence[bytes], features: Sequence[Features]) -> int:
        """Write new vectors as one shard; returns the bytes written (0 when skipped)."""
        from safetensors.torch import save_file

        if not keys:
            return 0
        tensors = {
            "keys": torch.tensor([list(k) for k in keys], dtype=torch.uint8),
            "offsets": torch.tensor([0] + [f.options.shape[0] for f in features], dtype=torch.int64).cumsum(0),
            "options": torch.cat([f.options for f in features]).to(CACHE_DTYPE).contiguous(),
            "summary": torch.stack([f.summary for f in features]).to(CACHE_DTYPE).contiguous(),
        }
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        current = _folder_bytes(self.path) if self.path.is_dir() else 0
        if current + size > self.max_bytes:
            self.log("feature cache: this set would take %s, more than cache_max_gb=%g; not writing it"
                     % (format_bytes(current + size), self.max_bytes / 1024**3))
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(self.root)
        if disk.free - size < MIN_FREE_FRACTION * disk.total:
            self.log("feature cache: writing %s would leave less than %d%% of the disk free; not writing it"
                     % (format_bytes(size), int(100 * MIN_FREE_FRACTION)))
            return 0
        self._evict(size)
        self.path.mkdir(exist_ok=True)
        meta = _read_meta(self.path) or {"format": FORMAT, "parts": self.parts, "created": time.time(),
                                         "n_questions": 0}
        n_shards = len(list(self.path.glob("shard-*.safetensors")))
        shard = self.path / ("shard-%05d.safetensors" % n_shards)
        tmp = shard.with_suffix(".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, shard)
        meta["n_questions"] += len(keys)
        self._touch(meta)
        return shard.stat().st_size

    def _evict(self, incoming: int) -> None:
        """Delete the least recently used other sets until `incoming` more bytes fit the cap."""
        others = sorted(((p, m) for p, m in _sets(self.root) if p != self.path),
                        key=lambda pm: pm[1].get("last_used", 0))
        total = _folder_bytes(self.root)
        while others and total + incoming > self.max_bytes:
            path, _ = others.pop(0)
            freed = _folder_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
            total -= freed
            self.log("feature cache: deleted the least recently used set %s (%s) to stay under cache_max_gb"
                     % (path.name, format_bytes(freed)))

    def folder_bytes(self) -> int:
        return _folder_bytes(self.root) if self.root.is_dir() else 0


def cache_info(cache_dir) -> list[dict]:
    """List the feature-cache sets in a ``cache_dir``.

    A model trained with ``tuning="head"`` and ``cache_dir`` writes the backbone's outputs there,
    grouped into sets: one per combination of base model, revision, kept layers, layout,
    precision, prompt format and token limits.

    Parameters
    ----------
    cache_dir : str or path
        The folder passed as ``cache_dir`` to the model.

    Returns
    -------
    list of dict
        One entry per set, most recently used first, with its ``set`` name, ``path``,
        ``model``, ``n_questions``, ``size_bytes``, a readable ``size``, ``last_used`` (ISO
        time) and the ``parts`` that identify it.

    See Also
    --------
    clear_cache : Delete the sets.
    """
    rows = []
    for path, meta in _sets(Path(cache_dir).expanduser()):
        size = _folder_bytes(path)
        last = meta.get("last_used")
        rows.append({
            "set": path.name,
            "path": str(path),
            "model": meta.get("parts", {}).get("model_name"),
            "n_questions": meta.get("n_questions", 0),
            "size_bytes": size,
            "size": format_bytes(size),
            "last_used": datetime.fromtimestamp(last, timezone.utc).isoformat(timespec="seconds") if last else None,
            "parts": meta.get("parts", {}),
        })
    return sorted(rows, key=lambda r: r["last_used"] or "", reverse=True)


def clear_cache(cache_dir) -> int:
    """Delete every feature-cache set in a ``cache_dir``.

    Other files in the folder are left alone. The cache also prunes itself: it stays under the
    model's ``cache_max_gb``, deleting the least recently used sets first.

    Parameters
    ----------
    cache_dir : str or path
        The folder passed as ``cache_dir`` to the model.

    Returns
    -------
    int
        The number of bytes freed.

    See Also
    --------
    cache_info : List the sets and their sizes.
    """
    freed = 0
    for path, _ in _sets(Path(cache_dir).expanduser()):
        freed += _folder_bytes(path)
        shutil.rmtree(path, ignore_errors=True)
    return freed
