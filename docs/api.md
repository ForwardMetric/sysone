# API reference

Everything below is importable from `sysone`.

## Models

```{eval-rst}
.. currentmodule:: sysone

.. autosummary::
   :toctree: generated
   :nosignatures:

   EncoderModel
   DecoderModel
   SystemOneModel
   NotFittedError
```

{class}`EncoderModel` and {class}`DecoderModel` take the same parameters, listed on each page,
and differ only in three defaults. {class}`SystemOneModel` is their shared base class.

## Evaluation

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   baselines
   compare
```

## Feature cache

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cache_info
   clear_cache
```

## Extending sysone

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Architecture
   register_architecture
   layout.Pieces
```

See [Adding a base model](user_guide/architectures.md) for a walkthrough.
