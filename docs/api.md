# API reference

Everything below is importable from `sysone`.

## Models

```{eval-rst}
.. currentmodule:: sysone

.. autosummary::
   :toctree: generated
   :nosignatures:

   SystemOneModel
   EncoderModel
   DecoderModel
   NotFittedError
```

`SystemOneModel(model_name)` returns whichever of {class}`EncoderModel` and
{class}`DecoderModel` suits the base model, and `SystemOneModel.load(path)` loads a saved model of
either kind. The two classes take the same parameters, listed on each page, and differ only in
three defaults.

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

## Serving

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   serve.create_app
```

The `sysone serve` command wraps it; see [Serving](user_guide/serving.md).

## Extending sysone

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Architecture
   register_architecture
   layout.Pieces
```

See [Base models](user_guide/architectures.md) for a walkthrough.
