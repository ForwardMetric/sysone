# Questions and data

Everything sysone reads is plain Python dicts, in the same shape whether it comes from a list,
a JSONL file or an API log. No sysone objects are needed to describe data.

## Questions

A question is a dict with a `type` and `instructions`. Choice and score questions also have
`criteria`.

```python
{"type": "choice", "instructions": "Which team should handle it?",
 "criteria": ["sales", "operations", "servicing"]}

{"type": "choice", "instructions": "Which team should handle it?",
 "criteria": {"sales": "quotes, pricing, renewals",          # labels with descriptions,
              "operations": "shipping and the warehouse",    # which the model reads too
              "servicing": "repairs, exchanges, returns"}}

{"type": "score", "instructions": "How severe is it?",
 "criteria": ["none", "minor", "major", "critical"]}         # levels, from low to high

{"type": "noul", "instructions": "Is the customer asking for a refund?"}

{"type": "noul", "instructions": "Is this spam?",
 "criteria": {"false": "a real customer message", "true": "unsolicited or automated"}}
```

| type | `criteria` | at least |
|---|---|---|
| `choice` | a list of labels, or `{label: description}` | 2 labels |
| `score` | a list of level descriptions, from low to high | 2 levels |
| `noul` | optional: `{"false": description, "true": description}` | |

The question's id (its key in the `questions` dict) is how answers are reported. The model reads
the instructions, the criteria and the option descriptions, so wording matters, and a question
worded differently from the training data is, to the model, a different question.

## Rows

A row is one piece of text, the `state`, with the questions to answer about it.

```python
{"state": "Hi, my order is stuck in the warehouse and our line is down.",
 "category": "email",
 "questions": {"team": {...}, "severity": {...}, "refund": {...}},
 "expected": {"team": "operations", "severity": 3, "refund": False}}
```

- **`state`**: a string, a dict (such as a ticket with fields), or a list of conversation turns.
  Anything that isn't a string is turned into JSON text.
- **`questions`**: `{id: question}`. Rows can carry different question sets: a sales row and an
  operations row needn't share anything but the `state` key.
- **`category`** (optional): a group such as a workflow, channel or customer. It is used to
  stratify the validation slice, for `balance="category"`, and for the per-category metrics of
  {meth}`~sysone.SystemOneModel.evaluate`.

## Labels

Each question in a training row needs exactly one label, given in one of three ways. Questions
without a label are skipped, and counted in `model.report_["skipped"]`.

**`expected` on the row**, a hard label per question:

| type | label | example |
|---|---|---|
| `choice` | one of the labels | `"operations"` |
| `score` | a level index (a fraction splits between two levels), or the level's text | `3`, `2.5` (half 2, half 3), `"critical"` |
| `noul` | a bool, or `"yes"` / `"no"` / `"true"` / `"false"` / `1` / `0` | `False` |

**`label` inside the question**, with the same meaning:

```python
{"state": "...", "questions": {"severity": {"type": "score", "instructions": "How severe?",
                                            "criteria": ["none", "minor", "major", "critical"],
                                            "label": 2}}}
```

**`gold` on the row**, a soft label: a probability per option, for example the share of raters
who chose each answer. Keys are the choice labels, `"false"` / `"true"`, or the level index as a
string. Missing options count as 0, and the probabilities are normalised.

```python
{"state": "...", "questions": {...},
 "gold": {"team": {"probabilities": {"operations": 0.7, "servicing": 0.3}},
          "severity": {"probabilities": {"2": 0.5, "3": 0.5}}}}
```

Soft labels keep the raters' disagreement in the training target instead of reducing it to a
majority vote, and evaluation scores them the same way as hard labels.

## Input forms

`fit`, `evaluate` and `baselines` accept any of these.

**Records**: a list (or any iterable, such as a `datasets.Dataset`) of rows as above.

**Records plus a question catalog**: define the questions once, and let rows name them by id
in `expected` or `gold`, or list them in `questions`:

```python
catalog = {"team": {...}, "severity": {...}, "refund": {...}}
rows = [{"state": "...", "expected": {"team": "sales", "refund": False}},
        {"state": "...", "questions": ["severity"], "expected": {"severity": 1}}]
model.fit(rows, questions=catalog)
```

**A long DataFrame**, one row per state and question, in one of two layouts:

| layout | columns |
|---|---|
| with a catalog | `state`, `question_id`, `answer`, optional `category`; pass `questions=catalog` |
| self-contained | `state`, `type`, `instructions`, `criteria` (a list or dict, or its JSON), `answer`, optional `category` |

**A file**: `.jsonl` or `.json` holding records, or `.csv` / `.parquet` holding a long table.
Reading `.csv` and `.parquet` needs pandas.

```python
model.fit("train.jsonl")
model.fit("train.csv", questions=catalog)
```

## How much data

There is no fixed minimum. The quickstart notebook learns its clear-cut synthetic questions
from 400 tickets; the example notebooks train on public datasets of thousands of rows. Compare
against {func}`~sysone.baselines` to see whether a model has learned more than the most common
answer. Calibration fits a separate temperature for a question type only when the
validation slice has at least 50 questions of that type; with less, probabilities are left as
the model produced them.
