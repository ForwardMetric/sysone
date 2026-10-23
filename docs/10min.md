# 10 minutes to sysone

A short tour for new users. It starts with a model that answers questions about support tickets,
then adds one situation at a time: fine-tuning, choosing what trains (LoRA, QLoRA, which layers,
the decision head), many texts at once, labels that disagree, data in files, knowing when not to
answer, picking a base model, long runs, saving and serving. Each step builds on the one before,
and every code block runs as is, in order.

The tour uses a tiny base model, [Ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m)
(about 70 MB), and synthetic tickets, so the whole tour runs in about four minutes on a
laptop CPU, after a one-time download. It passes
`device="cpu"` so that, with `seed=0`, you get the numbers shown here: some PyTorch operations on
Apple's GPU (MPS) aren't deterministic, so the same seed can give different results there. Leave
`device` out to use your GPU. For real work, use a larger base model on a GPU;
[Base models](user_guide/architectures.md) lists them.

```bash
pip install "sysone[serve]"
```

## Questions

A question is a dict. sysone has three types:

- **choice**: pick one label from a list;
- **score**: pick a level on an ordered scale, from low to high;
- **noul**: yes or no.

```python
from pprint import pprint

questions = {
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": ["sales", "operations", "servicing"]},
    "severity": {"type": "score", "instructions": "How severe is it?",
                 "criteria": ["none", "minor", "major", "critical"]},
    "refund": {"type": "noul",
               "instructions": "Is the customer asking for a refund?"},
}
```

## Labelled examples

A row is a piece of text (the `state`), the questions about it, and the right answers
(`expected`): a label for choice, a level index for score, a bool for noul. The tickets here are
generated, and like real data, not every one has a clear answer. `votes` keeps the share of
reviewers who picked each team:

- most tickets have a clear topic, and everyone agrees;
- about one in five has a topic reviewers disagree on, 70% to 30% (is a damaged delivery for
  operations or servicing?);
- about one in ten is vague, and reviewers split evenly.

`expected` holds the majority vote, or a random pick when there is none.

```python
import random

TOPICS = {
    "sales": ["a quote", "pricing", "a contract renewal", "a discount"],
    "operations": ["a late delivery", "a tracking number", "the warehouse",
                   "a missing parcel"],
    "servicing": ["a broken unit", "a repair", "an exchange", "a faulty part"],
}
DISPUTED = {
    "a damaged delivery": {"operations": 0.7, "servicing": 0.3},
    "an invoice for a repair": {"servicing": 0.7, "sales": 0.3},
    "a delayed contract start": {"sales": 0.7, "operations": 0.3},
}
VAGUE = ["my account", "an earlier email", "my last order"]
LEVELS = ["just a question", "a small issue", "a serious problem",
          "an emergency, everything is down"]
REFUNDS = ["Please refund me.", "I want my money back.", "Can I get a refund?"]


def ticket(rng):
    kind = rng.random()
    if kind < 0.7:      # a clear topic: everyone agrees
        team = rng.choice(list(TOPICS))
        topic, votes = rng.choice(TOPICS[team]), {team: 1.0}
    elif kind < 0.9:    # a disputed topic: 70% against 30%
        topic = rng.choice(list(DISPUTED))
        votes = DISPUTED[topic]
    else:               # a vague one: an even split
        topic, votes = rng.choice(VAGUE), {t: 1 / 3 for t in TOPICS}
    if max(votes.values()) > 0.5:
        team = max(votes, key=votes.get)     # the majority vote
    else:
        team = rng.choice(list(votes))       # no majority: a random pick
    level = rng.randrange(len(LEVELS))
    refund = rng.random() < 0.4
    text = "Hello, this is about %s. It is %s." % (topic, LEVELS[level])
    if refund:
        text += " " + rng.choice(REFUNDS)
    return {"state": text, "questions": questions, "votes": votes,
            "expected": {"team": team, "severity": level, "refund": refund}}


rng = random.Random(0)
train = [ticket(rng) for _ in range(600)]
held_out = [ticket(rng) for _ in range(300)]
test = [ticket(rng) for _ in range(300)]
pprint(train[0])
```

```text
{'expected': {'refund': True, 'severity': 0, 'team': 'servicing'},
 'questions': {'refund': {'instructions': 'Is the customer asking for a '
                                          'refund?',
                          'type': 'noul'},
               'severity': {'criteria': ['none', 'minor', 'major', 'critical'],
                            'instructions': 'How severe is it?',
                            'type': 'score'},
               'team': {'criteria': ['sales', 'operations', 'servicing'],
                        'instructions': 'Which team should handle it?',
                        'type': 'choice'}},
 'state': 'Hello, this is about an invoice for a repair. It is just a '
          'question. Can I get a refund?',
 'votes': {'sales': 0.3, 'servicing': 0.7}}
```

Extra keys such as `votes` are ignored by sysone; they are kept here for soft labels, later.

## A first model

{class}`~sysone.SystemOneModel` takes the name of a base model on the Hugging Face Hub and builds
the right kind of model for it. `tuning="head"` trains only a small decision head on top of the
frozen base model, which takes seconds.

```python
from sysone import SystemOneModel

model = SystemOneModel(
    "jhu-clsp/ettin-encoder-17m",
    tuning="head",
    device="cpu", seed=0,
)
model.fit(train)
print(type(model).__name__)
```

```text
[sysone] loaded jhu-clsp/ettin-encoder-17m (modernbert, 7 of 7 layers, float32 on cpu); tuning=head: head only; max_state_tokens=24, max_option_tokens=3
[sysone] 1620 training questions, 180 validation questions
[sysone] training from scratch: up to 20 epochs
[sysone] computing backbone features once for 1620 questions
[sysone] epoch 1/20  step 26  train_loss=1.1191  val_loss=1.1425  val_accuracy=0.5778
[sysone] epoch 2/20  step 52  train_loss=1.0909  val_loss=1.0667  val_accuracy=0.3778
...
[sysone] epoch 9/20  step 234  train_loss=0.4637  val_loss=0.7641  val_accuracy=0.7944
[sysone] early stopping: no lower validation loss in 3 epochs
[sysone] calibration: temperatures choice 1.61, score 0.963, noul 0.5; validation ECE 0.1310 -> 0.1009
EncoderModel
```

`fit` holds back 10% of the rows to decide when to stop and to calibrate the probabilities, and
prints its progress. Ask it about a new ticket:

```python
ticket_text = ("Hello, this is about a missing parcel. "
               "It is an emergency, everything is down. I want my money back.")
pprint(model.predict(ticket_text, questions))
```

```text
{'answers': {'refund': {'answer_confidence': 1.0,
                        'noul': 1.0,
                        'probabilities': {'false': 0.0, 'true': 1.0},
                        'type': 'noul'},
             'severity': {'answer_confidence': 0.8636,
                          'confidence': 0.8199,
                          'legend': {'0': 'none',
                                     '1': 'minor',
                                     '2': 'major',
                                     '3': 'critical'},
                          'probabilities': {'0': 0.0117,
                                            '1': 0.0202,
                                            '2': 0.1044,
                                            '3': 0.8636},
                          'score': 2.8199,
                          'type': 'score'},
             'team': {'answer_confidence': 0.9169,
                      'choice': 'operations',
                      'confidence': 0.8753,
                      'probabilities': {'operations': 0.9169,
                                        'sales': 0.013,
                                        'servicing': 0.0702},
                      'type': 'choice'}},
 'usage': {'input_tokens': 156, 'output_tokens': 0}}
```

Every answer has probabilities. `answer_confidence` is how likely the answer is right, and after
calibration it means what it says: answers given with 0.9 are right about 90% of the time.

## Is it any good?

{meth}`~sysone.SystemOneModel.evaluate` scores a model on labelled rows it hasn't seen.
{func}`~sysone.baselines` scores two simple guessers to compare with: a random guess, and the
average answer in the training data. {func}`~sysone.compare` puts the reports side by side.

```python
from sysone import baselines, compare

results = baselines(test, train=train)
results["head"] = model.evaluate(test)
print(compare(results))                       # accuracy per question type
```

```text
                  choice     score      noul   overall
random guess    0.333333  0.250000  0.500000  0.361111
average answer  0.320000  0.233333  0.586667  0.380000
head            0.740000  0.473333  0.993333  0.735556
```

Training only the head beats both guessers, but it leaves room, especially on severity.

## Fine-tuning

Training the base model's layers too (`tuning="full"`) is slower and more accurate. On a real
dataset and a larger model, do this on a GPU.

```python
full = SystemOneModel(
    "jhu-clsp/ettin-encoder-17m",
    tuning="full", epochs=4,
    device="cpu", seed=0, verbose=0,
)
full.fit(train)

results["full"] = full.evaluate(test)
print(compare(results, by="overall")[["accuracy", "log_loss", "brier", "ece"]])
```

```text
                accuracy  log_loss     brier       ece
random guess    0.361111  1.059351  0.638889  0.000000
average answer  0.380000  1.055855  0.634846  0.026111
head            0.735556  0.579557  0.316315  0.084550
full            0.972222  0.054030  0.031098  0.009664
```

Fully tuned, accuracy rises to 0.97, and log loss, Brier score and calibration error (`ece`) all
fall: the probabilities are sharper and still honest.

`verbose=0` keeps `fit` quiet. `full.history_` keeps every validation check, for learning curves:

```python
for check in full.history_[-3:]:
    print(check["epoch"], round(check["val_loss"], 4),
          round(check["val_accuracy"], 4))
```

```text
3.808 0.0734 0.9667
3.923 0.0737 0.9667
4.0 0.0737 0.9667
```

## Choosing what trains

`tuning` is the main decision: how much of the base model learns.

| `tuning` | what trains | use it for |
|---|---|---|
| `"head"` | only the decision head; the base model is frozen and its outputs are cached | a first check of the data and questions: seconds, even on a laptop |
| `"lora"` | small adapter matrices on the top layers; the base model's own weights stay frozen | most of full tuning's accuracy in less memory |
| `"qlora"` | the same adapters, with the frozen weights stored in 4 bits | the least memory, for large decoders; needs an NVIDIA GPU |
| `"full"` | every layer | the best accuracy when there is enough data; encoders' default |

Try LoRA. With `verbose` on, the first line of `fit` says what trains:

```python
lora = SystemOneModel(
    "jhu-clsp/ettin-encoder-17m",
    tuning="lora", epochs=4,
    device="cpu", seed=0,
)
lora.fit(train)
results["lora"] = lora.evaluate(test)
```

```text
[sysone] loaded jhu-clsp/ettin-encoder-17m (modernbert, 7 of 7 layers, float32 on cpu); tuning=lora: adapters on 4 layers; max_state_tokens=24, max_option_tokens=3
[sysone] 1620 training questions, 180 validation questions
[sysone] learning rates: head 0.001, adapters 0.001
[sysone] training from scratch: up to 4 epochs
...
[sysone] epoch 4/4  step 104  train_loss=0.0576  val_loss=0.1320  val_accuracy=0.9611
[sysone] calibration: temperatures choice 0.967, score 0.5, noul 0.5; validation ECE 0.0392 -> 0.0311
```

By default LoRA puts adapters on the top 60% of the layers, here 4 of 7. Without an NVIDIA
GPU, `tuning="qlora"` trains as `"lora"` and says so in a warning, so the same code runs on a
laptop and on a GPU server.

**Which layers.** Three settings say which layers of the base model are used and how they learn.
Each takes a count, a fraction of the layers, or `"all"`:

- `num_layers`: keep only the bottom layers; the layers above are never loaded, which saves
  memory and time;
- `trainable_layers`: train the top layers fully;
- `lora_layers`: put adapters on the layers below those; `lora_rank` sets the adapters' size.

**The decision head.** `head_layers` (2 by default) and `head_dim` (half the base model's width)
set its size, `head_dropout` its dropout. `backbone_lr` and `head_lr` set the learning rates;
sysone picks `backbone_lr` from the model family and what trains.

Compare a few setups:

```python
setups = {
    "lora on all layers, rank 16": dict(tuning="lora", lora_layers="all",
                                        lora_rank=16),
    "top layer full, lora on 3 below": dict(tuning="lora", trainable_layers=1,
                                            lora_layers=3),
    "bottom 5 layers, top 2 trained": dict(tuning="full", num_layers=5,
                                           trainable_layers=2),
    "full, with a larger head": dict(tuning="full", head_layers=4, head_dim=256),
}
for name, settings in setups.items():
    candidate = SystemOneModel(
        "jhu-clsp/ettin-encoder-17m", epochs=4,
        device="cpu", seed=0, verbose=0, **settings,
    )
    results[name] = candidate.fit(train).evaluate(test)

shown = {name: results[name] for name in ["full", "lora", *setups]}
print(compare(shown, by="overall")[["accuracy", "log_loss"]])
```

```text
                                 accuracy  log_loss
full                             0.972222  0.054030
lora                             0.971111  0.085526
lora on all layers, rank 16      0.975556  0.059830
top layer full, lora on 3 below  0.942222  0.124640
bottom 5 layers, top 2 trained   0.707778  0.616696
full, with a larger head         0.967778  0.065251
```

On this small dataset LoRA comes close to full tuning, adapters on every layer help a little,
a larger head doesn't, and dropping the top layers costs the most. Which setup wins depends on the base model and the
data, so compare on yours. [Choosing how to train](user_guide/training.md) has the details.

## Many texts, and other questions

Pass a list of texts to answer them together, which is much faster than one at a time. Questions
are given at prediction time, so you can ask any subset:

```python
texts = [
    "Hello, this is about pricing for 40 more seats. It is just a question.",
    "Hello, this is about a missing parcel. "
    "It is an emergency, everything is down.",
]
for result in full.predict(texts, {"team": questions["team"]}):
    team = result["answers"]["team"]
    print(team["choice"], team["answer_confidence"])
```

```text
sales 0.9993
operations 1.0
```

Rows can also carry their own questions, the same shape as training rows without labels:

```python
rows = [{"state": texts[0], "questions": {"refund": questions["refund"]}},
        {"state": texts[1], "questions": {"severity": questions["severity"]}}]
for result in full.predict(rows):
    pprint(result["answers"])
```

```text
{'refund': {'answer_confidence': 1.0,
            'noul': 0.0,
            'probabilities': {'false': 1.0, 'true': 0.0},
            'type': 'noul'}}
{'severity': {'answer_confidence': 1.0,
              'confidence': 1.0,
              'legend': {'0': 'none',
                         '1': 'minor',
                         '2': 'major',
                         '3': 'critical'},
              'probabilities': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 1.0},
              'score': 3.0,
              'type': 'score'}}
```

## When people disagree: soft labels

On a disputed ticket, 30% of reviewers disagree with the majority, but the hard label keeps
only the majority vote, so a model trained on it learns to be certain. A **soft
label** keeps the split: a probability per option, here the share of reviewers. Put it in
`gold`. A question takes its label from one place, so `team` moves out of `expected`, and the
other questions keep their hard labels:

```python
def soft(row):
    expected = {q: a for q, a in row["expected"].items() if q != "team"}
    gold = {"team": {"probabilities": row["votes"]}}
    return {**row, "expected": expected, "gold": gold}


soft_model = SystemOneModel(
    "jhu-clsp/ettin-encoder-17m",
    tuning="full", epochs=4,
    device="cpu", seed=0, verbose=0,
)
soft_model.fit([soft(r) for r in train])

disputed = "Hello, this is about a damaged delivery. It is a small issue."
for name, m in [("hard labels", full), ("soft labels", soft_model)]:
    answer = m.predict(disputed, {"team": questions["team"]})["answers"]["team"]
    print(name, answer["probabilities"])
```

```text
hard labels {'sales': 0.0, 'operations': 1.0, 'servicing': 0.0}
soft labels {'sales': 0.0059, 'operations': 0.7548, 'servicing': 0.2393}
```

The hard-label model is certain; the soft-label model learned the reviewers' 70/30 split. Scored
against the reviewers' split on all the test tickets, the soft-label model is closer (`distance`
is 0 for identical distributions, 1 for completely different):

```python
soft_test = [soft(r) for r in test]
reports = {"hard labels": full.evaluate(soft_test),
           "soft labels": soft_model.evaluate(soft_test)}
print(compare(reports, by="type", metric="distance"))
```

```text
               choice     score          noul   overall
hard labels  0.080953  0.000004  1.007296e-06  0.026986
soft labels  0.025330  0.000001  2.135050e-08  0.008444
```

`evaluate` scores soft labels the same way as hard ones, and accuracy counts the reviewers' top
choice as right.

## Data in files and tables

`fit`, `evaluate` and `baselines` also take a JSONL, JSON, CSV or Parquet file, or a pandas
DataFrame. With a **question catalog**, rows name questions by id instead of repeating them:

```python
import json

with open("tickets.jsonl", "w") as f:
    for r in train:
        record = {"state": r["state"], "expected": r["expected"]}
        f.write(json.dumps(record) + "\n")

from_file = SystemOneModel(
    "jhu-clsp/ettin-encoder-17m",
    tuning="head",
    device="cpu", seed=0, verbose=0,
)
from_file.fit("tickets.jsonl", questions=questions)
print(from_file.predict(texts[1], questions)["answers"]["team"]["choice"])
```

```text
operations
```

A long table, one row per text and question, works the same way:

```python
import pandas as pd

table = pd.DataFrame([{"state": r["state"], "question_id": q, "answer": a}
                      for r in train for q, a in r["expected"].items()])
print(table.head(3))
from_table = SystemOneModel(
    "jhu-clsp/ettin-encoder-17m",
    tuning="head",
    device="cpu", seed=0, verbose=0,
)
from_table.fit(table, questions=questions)
print(from_table.predict(texts[1], questions)["answers"]["team"]["choice"])
```

```text
                                               state question_id     answer
0  Hello, this is about an invoice for a repair. ...        team  servicing
1  Hello, this is about an invoice for a repair. ...    severity          0
2  Hello, this is about an invoice for a repair. ...      refund       True
operations
```

## Knowing when not to answer

A system one model is most useful when it handles the easy cases and hands the rest to a person
or a larger model. {meth}`~sysone.SystemOneModel.abstention_threshold` picks, per question type,
the lowest `answer_confidence` at which answers reach a target accuracy, on rows held out from
training:

```python
thresholds = full.abstention_threshold(held_out, target_accuracy=0.99)
pprint(thresholds["choice"])
```

```text
{'accuracy': 0.9965034965034965,
 'coverage': 0.9533333333333334,
 'n': 300,
 'threshold': 0.5204929113388062}
```

There is one threshold per question type. Pass them as `abstain`, and each answer below its
threshold is marked `"abstained": True` (it still carries its answer and probabilities). The
model holds back on the vague tickets, where reviewers split three ways, and answers the rest:

```python
answers = full.predict([r["state"] for r in test], {"team": questions["team"]},
                       abstain=thresholds)
held_back = {}
for result, row in zip(answers, test):
    top_share = max(row["votes"].values())
    kind = {1.0: "clear", 0.7: "disputed"}.get(top_share, "vague")
    counts = held_back.setdefault(kind, [0, 0])
    counts[0] += result["answers"]["team"]["abstained"]
    counts[1] += 1
print({kind: "%d of %d abstained" % tuple(counts)
       for kind, counts in held_back.items()})
```

```text
{'clear': '0 of 207 abstained', 'vague': '24 of 36 abstained', 'disputed': '0 of 57 abstained'}
```

`evaluate` with `abstain` reports `coverage`, the share answered, and scores only the answered
questions:

```python
report = full.evaluate(test, abstain=thresholds)
reports = {"answer all": full.evaluate(test), "abstain": report}
print(compare(reports, by="type", metric="accuracy"))
print(compare({"abstain": report}, by="type", metric="coverage"))
```

```text
              choice  score  noul   overall
answer all  0.916667    1.0   1.0  0.972222
abstain     0.971014    1.0   1.0  0.990826
         choice  score      noul   overall
abstain    0.92    1.0  0.986667  0.968889
```

## Choosing the base model

`SystemOneModel` works with encoders (such as ModernBERT, Ettin, mmBERT and EuroBERT) and
decoders (such as Qwen3, Llama, Mistral and Gemma 3). It builds an
{class}`~sysone.EncoderModel` or a {class}`~sysone.DecoderModel`, whose defaults suit each kind:
encoders train every layer, while decoders keep the bottom 70% of their layers and train small
adapters on them (QLoRA).

```python
from sysone import DecoderModel, EncoderModel

print(type(SystemOneModel("jhu-clsp/ettin-encoder-17m")).__name__)
print(type(SystemOneModel("Qwen/Qwen3-0.6B")).__name__)
```

```text
EncoderModel
DecoderModel
```

Nothing is downloaded until `fit`. You can use the two classes directly; using the wrong one for
a base model raises an error that names the right one:

```python
try:
    EncoderModel("Qwen/Qwen3-0.6B").fit(train)
except ValueError as error:
    print(error)
```

```text
Qwen/Qwen3-0.6B is a decoder (qwen3), but EncoderModel is for encoders; use DecoderModel, or SystemOneModel to pick the class from the base model
```

## Long runs: autosave and resume

Training on a large dataset can take hours, and a notebook can disconnect. With `checkpoint_dir`,
`fit` saves a usable model and its training state every `checkpoint_every` optimizer steps, and
when it finishes. Load the folder and call `fit` again to continue a run that stopped early:

```python
run = SystemOneModel(
    "jhu-clsp/ettin-encoder-17m",
    tuning="full", epochs=2,
    checkpoint_dir="ticket-run", checkpoint_every=20,
    device="cpu", seed=0, verbose=0,
)
run.fit(train)

resumed = SystemOneModel.load("ticket-run", verbose=1)   # show progress
resumed.set_params(epochs=3)   # one more epoch than the saved run: it continues
resumed.fit(train)
```

```text
[sysone] 1620 training questions, 180 validation questions
[sysone] learning rates: head 0.001, fully trained layers 5e-05
[sysone] resuming at epoch 2 of 3 (step 52)
[sysone] autosaving to ticket-run when training ends or is interrupted, and every 20 steps
[sysone] epoch 2.077/3  step 54  train_loss=0.3623  val_loss=0.5046  val_accuracy=0.8111
[sysone] epoch 2.192/3  step 57  train_loss=0.4067  val_loss=0.5215  val_accuracy=0.7056
...
[sysone] epoch 3/3  step 78  train_loss=0.1827  val_loss=0.2510  val_accuracy=0.9222
[sysone] calibration: temperatures choice 0.59, score 0.5, noul 0.505; validation ECE 0.0820 -> 0.0842
```

## Save, load and share

```python
full.save("ticket-model")
same = SystemOneModel.load("ticket-model", verbose=0)
print(same.predict(texts[1], questions) == full.predict(texts[1], questions))
```

```text
True
```

The folder holds what was trained; the base model itself is downloaded again by name.
`model.push_to_hub("your-name/ticket-model")` uploads it with a model card, and
`SystemOneModel.load("your-name/ticket-model")` loads it anywhere; see
[Saving and sharing](user_guide/saving.md).

## Serving it as an API

`sysone serve` turns a saved model into an HTTP API, with the request and response shape of
TypeSafe's Jev System One API:

```bash
sysone serve ticket-model
```

```text
sysone 0.5.2 serving ticket-model at http://127.0.0.1:8000/v1/systemone
```

From another terminal:

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Hello, this is about a contract renewal. It is a small issue. Please refund me.",
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle it?",
                         "criteria": ["sales", "operations", "servicing"]},
                "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}
}'
```

```text
{"id": "dec-af13d9477a0147d892a40bf4", "model": "ticket-model",
 "answers": {
   "team": {"type": "choice", "choice": "sales",
            "probabilities": {"sales": 1.0, "operations": 0.0, "servicing": 0.0},
            "confidence": 1.0, "answer_confidence": 1.0},
   "refund": {"type": "noul", "noul": 1.0, "probabilities": {"false": 0.0, "true": 1.0},
              "answer_confidence": 1.0}},
 "usage": {"input_tokens": 87, "output_tokens": 0}}
```

The answers are what `predict` returns, so the same code handles both.

See [Serving](user_guide/serving.md) for API keys, abstention and deployment.

## Where next

- [Quickstart](quickstart.md) and the [example notebooks](examples.md), which train on real
  datasets.
- [Choosing how to train](user_guide/training.md): tuning, early stopping and the training
  objective.
- [Questions and data](user_guide/data.md), [Predictions](user_guide/predictions.md) and
  [Evaluation and calibration](user_guide/evaluation.md) in full.
