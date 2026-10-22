# Development

## Setup

sysone uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ForwardMetric/sysone
cd sysone
uv sync
```

## Tests and linting

```bash
uv run pytest            # fast tests on tiny random-weight models; no downloads, CPU only
uv run pytest -m slow    # end-to-end tests on the real base models (~1.8 GB download)
uv run ruff check .
```

The fast tests run every architecture × tuning × layout combination on tiny models built in the
test session. Tests that need CUDA (fp16 loss scaling, 4-bit loading) are skipped elsewhere.

CI runs ruff and the fast tests on every pull request, with the newest dependencies on Python
3.12 and the declared minimum versions on Python 3.10, and builds these docs.

## Documentation

```bash
uv run --group docs sphinx-build -W docs docs/_build/html
open docs/_build/html/index.html
```

`-W` turns warnings into errors, as in CI. The API reference is generated from the docstrings,
which use the [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html). The
example notebooks are copied in from `examples/` and rendered without running them.

## Design and milestones

[DESIGN.md](DESIGN.md) is the source of truth for how sysone works and what each milestone
delivers. A change to behaviour starts as a change to the design.

## Releasing

1. Bump `src/sysone/version.py` and add the release to [the changelog](changelog.md).
2. Point the notebooks' install cell at the new minimum version (`sysone>=X.Y`).
3. Merge to `main` once CI passes.
4. Build and upload: `uv build && uvx twine upload dist/*`.
5. Tag the release: `git tag -a vX.Y.Z -m "sysone X.Y.Z" && git push origin vX.Y.Z`.
