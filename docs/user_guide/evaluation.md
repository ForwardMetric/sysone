# Evaluation and calibration

## Scoring a model

{meth}`~sysone.SystemOneModel.evaluate` scores a model on labelled data, in any form `fit`
accepts:

```python
report = model.evaluate(test_rows)
report["overall"]                   # {"n": 1200, "accuracy": 0.81, "log_loss": 0.52, ...}
report["by_type"]["score"]          # per question type
report["by_category"]["email"]      # per category
report["by_question"]["email/team"] # per question, keyed category/id
```

Hard and soft labels are scored the same way: each label becomes a target distribution over the
question's options, and the metrics compare the predicted distribution with it.

| metric | what it measures | better |
|---|---|---|
| `accuracy` | the top predicted option is the right one (for soft labels, the raters' top choice). Ties share the credit, so a uniform guess scores exactly 1/k. | higher |
| `distance` | total variation distance to the label: 0 identical, 1 completely different | lower |
| `log_loss` | cross-entropy against the label | lower |
| `brier` | squared distance to the label, summed over options | lower |
| `ece` | expected calibration error of the top option: how far "says X% sure" is from "right X% of the time" | lower |
| `score_error` | score questions: the gap between predicted and true expected level, in levels | lower |
| `auc` | yes/no questions: how well P(yes) ranks yes above no; 0.5 is chance | higher |

## Comparing with baselines

A model is only useful if it beats simple guessers. {func}`~sysone.baselines` scores two:

- **random guess**: every option equally likely;
- **average answer**: for each question, the average label in the training data. On data where
  one answer dominates, this alone can score high accuracy.

{func}`~sysone.compare` puts reports side by side:

```python
from sysone import baselines, compare

results = baselines(test_rows, train=train_rows)
results["head"] = head_model.evaluate(test_rows)
results["full"] = full_model.evaluate(test_rows)

compare(results)                                      # accuracy per question type
compare(results, by="overall")                        # every metric
compare(results, by="question", metric="score_error") # one metric per question
```

## Calibration

A model's raw probabilities are usually over-confident: answers given with 0.95 are right less
often than 95% of the time. After training, sysone fits a **temperature** on the validation
slice, a single number that softens (or sharpens) the probabilities until they match how often
answers are right. It never changes which option is on top, so accuracy is unchanged; `log_loss`,
`brier`, `ece` and `answer_confidence` change.

- A temperature is fitted per question type (choice, score, noul), and also per number of
  options (2, 3–5, 6–10, 11 or more) where the validation slice has at least 50 such questions.
  With fewer than 50 questions of a type, its temperature stays at 1.0.
- Temperatures are kept between 0.5 and 5, so a small validation slice can't push them to
  extremes.
- `model.report_["calibration"]` records the temperatures and the validation ECE before and
  after.
- `calibrate=False` turns it off.

To see what calibration changed on test data:

```python
results = {"calibrated": model.evaluate(test_rows),
           "raw": model.evaluate(test_rows, calibrated=False)}
compare(results, by="overall")[["accuracy", "log_loss", "brier", "ece"]]
```

## The training report

After `fit`, `model.report_` summarises the run: the number of training and validation
questions, what was skipped and why, the loss and accuracy on the validation slice, and the
calibration summary. `model.history_` has one record per validation check, for plotting
learning curves.
