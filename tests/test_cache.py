"""The on-disk feature cache: reuse, growth, invalidation, the size cap and the free-space guard."""

import shutil
from collections import namedtuple

import pytest
import torch

import sysone.base
from sysone import DecoderModel, EncoderModel, cache_info, clear_cache
from sysone.cache import FeatureCache, question_key
from sysone.features import Features
from sysone.layout import Encoded

from .conftest import ENCODERS, QUESTIONS, make_rows

FAST = dict(tuning="head", epochs=2, batch_size=16, precision="fp32", device="cpu", verbose=0)


def _model(kind, path, **params):
    klass = EncoderModel if kind in ENCODERS else DecoderModel
    return klass(path, **{**FAST, **params})


def _count_extract(monkeypatch):
    """Count the questions the backbone actually runs on during fit."""
    seen = {"n": 0}
    extract = sysone.base.extract

    def counting(arch, backbone, encoded, *args, **kwargs):
        seen["n"] += len(encoded)
        return extract(arch, backbone, encoded, *args, **kwargs)

    monkeypatch.setattr(sysone.base, "extract", counting)
    return seen


def test_second_fit_reads_features_from_disk(tiny_model, tmp_path, monkeypatch):
    kind, path = tiny_model
    rows = make_rows(30)
    seen = _count_extract(monkeypatch)
    first = _model(kind, path, cache_dir=tmp_path / "c").fit(rows)
    assert seen["n"] == 45
    [info] = cache_info(tmp_path / "c")
    assert info["n_questions"] == 45 and info["size_bytes"] > 0 and info["model"] == path

    seen["n"] = 0
    second = _model(kind, path, cache_dir=tmp_path / "c").fit(rows)
    assert seen["n"] == 0
    assert second.predict("red refund", QUESTIONS) == first.predict("red refund", QUESTIONS)
    no_cache = _model(kind, path).fit(rows)
    assert no_cache.predict("red refund", QUESTIONS) == first.predict("red refund", QUESTIONS)


def test_growing_data_computes_only_new_questions(tiny_models, tmp_path, monkeypatch):
    path = tiny_models["qwen3"]
    seen = _count_extract(monkeypatch)
    # the same token limits, so the two fits share a cache set
    _model("qwen3", path, cache_dir=tmp_path, max_state_tokens=32, max_option_tokens=8).fit(make_rows(20))
    seen["n"] = 0
    _model("qwen3", path, cache_dir=tmp_path, max_state_tokens=32, max_option_tokens=8).fit(make_rows(30))
    assert seen["n"] == 15  # rows 20-29: ten rows, 1.5 questions each
    [info] = cache_info(tmp_path)
    assert info["n_questions"] == 45


def test_any_key_change_starts_a_new_set(tiny_models, tmp_path):
    path = tiny_models["qwen3"]
    _model("qwen3", path, cache_dir=tmp_path).fit(make_rows(10))
    _model("qwen3", path, cache_dir=tmp_path, layout="sequential", shuffle_options=None).fit(make_rows(10))
    _model("qwen3", path, cache_dir=tmp_path, num_layers=2).fit(make_rows(10))
    sets = cache_info(tmp_path)
    assert len(sets) == 3
    assert {s["parts"]["layout"] for s in sets} == {"parallel", "sequential"}
    assert clear_cache(tmp_path) == sum(s["size_bytes"] for s in sets)
    assert cache_info(tmp_path) == []


def _features(n, hidden=64):
    return [Features(torch.randn(3, hidden).bfloat16(), torch.randn(hidden).bfloat16()) for _ in range(n)]


def _keys(n, salt=0):
    return [question_key(Encoded([i, salt], [0, 1], [0, 0], [0], 1)) for i in range(n)]


def test_cap_evicts_least_recently_used_sets(tmp_path):
    messages = []
    one = 30 * 4 * 64 * 2  # about one set of 30 questions
    sets = [FeatureCache(tmp_path, {"model_name": "m%d" % i}, 2.5 * one, messages.append) for i in range(3)]
    for i, cache in enumerate(sets):
        assert cache.add(_keys(30, i), _features(30)) > 0
    names = [s["parts"]["model_name"] for s in cache_info(tmp_path)]
    assert names == ["m2", "m1"]
    assert any("least recently used" in m and sets[0].path.name in m for m in messages)
    assert sets[1].lookup(_keys(30, 1))  # m1 is now the most recent
    FeatureCache(tmp_path, {"model_name": "m3"}, 2.5 * one, messages.append).add(_keys(30, 3), _features(30))
    assert [s["parts"]["model_name"] for s in cache_info(tmp_path)] == ["m3", "m1"]


def test_set_larger_than_the_cap_is_never_written(tmp_path):
    messages = []
    cache = FeatureCache(tmp_path, {"model_name": "m"}, 1000, messages.append)
    assert cache.add(_keys(30), _features(30)) == 0
    assert cache_info(tmp_path) == [] and "more than cache_max_gb" in messages[0]


def test_free_space_guard(tmp_path, monkeypatch):
    messages = []
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(10**9, 10**9 - 10**8 - 1000, 10**8 + 1000))
    cache = FeatureCache(tmp_path, {"model_name": "m"}, 10**9, messages.append)
    assert cache.add(_keys(30), _features(30)) == 0
    assert "less than 10% of the disk free" in messages[0]


def test_lookup_returns_what_was_written(tmp_path):
    cache = FeatureCache(tmp_path, {"model_name": "m"}, 10**9, print)
    feats = _features(5)
    cache.add(_keys(5), feats)
    found = cache.lookup(_keys(7))
    assert len(found) == 5
    for key, f in zip(_keys(5), feats):
        assert torch.equal(found[key].options, f.options) and torch.equal(found[key].summary, f.summary)


def test_cache_dir_is_ignored_when_the_backbone_learns(tiny_models, tmp_path):
    with pytest.warns(UserWarning, match="cache_dir is ignored"):
        _model("qwen3", tiny_models["qwen3"], tuning="lora", epochs=1, cache_dir=tmp_path).fit(make_rows(10))
    assert cache_info(tmp_path) == []
