# Predictions

{meth}`~sysone.SystemOneModel.predict` answers questions about one text or many. Questions are
given at prediction time, so one model serves any set of questions, although it answers best on
questions like those it was trained on.

## Calling `predict`

```python
model.predict(state, questions)           # one text -> one result
model.predict([s1, s2, s3], questions)    # many texts, same questions -> a list of results
model.predict(rows)                       # rows that carry their own questions -> a list
```

`rows` have the same shape as training rows, without labels: `{"state": ..., "questions": {...}}`.
A state can be a string, a dict, or a list of conversation turns; a list of turns must be
wrapped so it isn't read as many states: `model.predict([turns], questions)`.

States longer than `max_state_tokens` (measured on the training data) keep their first
`max_state_tokens` tokens.

## The result

```python
{"answers": {
    "team":     {"type": "choice", "choice": "servicing",
                 "probabilities": {"sales": 0.03, "operations": 0.05, "servicing": 0.92},
                 "confidence": 0.88, "answer_confidence": 0.92},
    "severity": {"type": "score", "score": 1.2,
                 "legend": {"0": "none", "1": "minor", "2": "major", "3": "critical"},
                 "probabilities": {"0": 0.1, "1": 0.62, "2": 0.24, "3": 0.04},
                 "confidence": 0.55, "answer_confidence": 0.62},
    "refund":   {"type": "noul", "noul": 0.97, "answer_confidence": 0.97}},
 "usage": {"input_tokens": 118, "output_tokens": 0}}
```

Results are plain dicts, ready to serialise as JSON, in the response shape of the [System One API](https://typesafe.ai) (Jev `/v1/systemone`). The answer fields depend on the type:

| field | choice | score | noul |
|---|---|---|---|
| the answer | `choice`: the most probable label | `score`: the expected level, `Σ level × p` | `noul`: the probability of yes |
| `probabilities` | per label | per level index | |
| `legend` | | level index to its text | |
| `confidence` | yes | yes | |
| `answer_confidence` | yes | yes | yes |

`score` is an average, so `1.2` means "mostly minor, some chance of major". The most likely level
is the one with the highest probability.

## Confidence

There are two measures, for different uses:

- **`answer_confidence`** is the probability of the top option after calibration. It answers
  "how likely is this answer right?": over many answers given with 0.9, about 90% are right.
  Use it to decide when to act on an answer and when to send it to a person.
- **`confidence`** measures how concentrated the distribution is, from 0 (no preference) to 1
  (certain), and is comparable across questions with different numbers of options:
  - choice: `(max p − 1/n) / (1 − 1/n)`, so a uniform guess gives 0 whatever the number of
    labels;
  - score: one minus the spread of probability around the most likely level, relative to the
    spread of a uniform distribution, so mass on neighbouring levels lowers it less than mass
    far away.

Calibration never changes which option is on top; see
[Evaluation and calibration](evaluation.md).

## Abstaining

A model can say "not sure" instead of guessing: pass `abstain`, a threshold on
`answer_confidence`, and each answer below it is marked `"abstained": True`. Send those to a
person or a larger model, and act on the rest.

```python
model.predict(text, questions, abstain=0.8)
# {"answers": {"team": {"type": "choice", "choice": "servicing", ..., "answer_confidence": 0.62,
#                       "abstained": True}, ...}, ...}

model.predict(text, questions, abstain={"choice": 0.8, "noul": 0.9})   # per question type
model.predict(text, questions, abstain={"team": 0.7, "choice": 0.8})   # a question id wins over its type
```

An abstained answer still carries its choice and probabilities, so the result keeps the same
shape and you can log what the model would have said. Without `abstain`, answers have no
`abstained` field.

Because `answer_confidence` is calibrated, a threshold means what it says: answers given with
0.9 or more are right about 90% of the time or better. To choose thresholds for a target
accuracy on your own data, see [Choosing abstention thresholds](evaluation.md#choosing-abstention-thresholds).

## Usage

`usage.input_tokens` counts the tokens the model read for all of a state's questions;
`output_tokens` is always 0, since nothing is generated.
