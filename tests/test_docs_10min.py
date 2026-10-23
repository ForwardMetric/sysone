"""Runs every Python block of docs/10min.md, in order, so the tour stays runnable. Downloads
Ettin-encoder-17m and takes about four minutes: run with `pytest -m slow`."""

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "docs" / "10min.md"


@pytest.mark.slow
def test_every_block_of_the_tour_runs(tmp_path, monkeypatch):
    blocks = re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S)
    assert len(blocks) > 10
    monkeypatch.chdir(tmp_path)  # the tour writes files and folders in the working directory
    namespace = {}
    for i, code in enumerate(blocks):
        exec(compile(code, "%s, block %d" % (PAGE.name, i), "exec"), namespace)
    assert namespace["same"].predict(namespace["texts"][1], namespace["questions"]) == \
        namespace["full"].predict(namespace["texts"][1], namespace["questions"])
