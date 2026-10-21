"""Train on a public typed-decision dataset from the Hugging Face Hub.

    uv run --with datasets python examples/public_dataset.py --model qwen --max-rows 5000

Datasets:
- `avbiswas/bev-decision-150K` (default): hard labels inside each question, a `domain` per row
  (used as the category), and a separate test split.
- `LocalLLaMA/typed-decisions`: soft `gold` probabilities per question.
"""

import argparse
from collections import defaultdict

from datasets import load_dataset

from sysone import DecoderModel, EncoderModel


def rows_from(dataset: str, split: str, max_rows: int):
    if dataset == "avbiswas/bev-decision-150K":
        ds = load_dataset(dataset, split=split).shuffle(seed=0).select(range(max_rows))
        return [{"state": r["state"], "questions": r["questions_json"], "category": r["domain"]} for r in ds]
    ds = load_dataset(dataset, "all", split=split).shuffle(seed=0).select(range(max_rows))
    return [{"state": r["state"], "questions": r["questions"], "gold": r["gold"], "category": r["workflow"]}
            for r in ds]


def accuracy_by_type(model, rows):
    """Accuracy per question type on rows whose questions carry a `label` (bev-decision-150K)."""
    import json

    hits, counts = defaultdict(int), defaultdict(int)
    for row, result in zip(rows, model.predict(rows)):
        questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        for qid, ans in result["answers"].items():
            label = questions[qid].get("label")
            if label is None:
                continue
            kind = ans["type"]
            if kind == "choice":
                hit = ans["choice"] == label
            elif kind == "score":
                hit = round(ans["score"]) == int(label)
            else:
                hit = (ans["noul"] > 0.5) == bool(label)
            hits[kind] += hit
            counts[kind] += 1
    return {kind: round(hits[kind] / counts[kind], 3) for kind in counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "modernbert"], default="qwen")
    parser.add_argument("--dataset", default="avbiswas/bev-decision-150K")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--tuning", default="head")
    args = parser.parse_args()

    klass = DecoderModel if args.model == "qwen" else EncoderModel
    model = klass(tuning=args.tuning).fit(rows_from(args.dataset, "train", args.max_rows))
    print("validation:", model.report_["validation"])
    if args.dataset == "avbiswas/bev-decision-150K":
        test = rows_from(args.dataset, "test", min(1000, args.max_rows))
        test_unlabeled = [{"state": r["state"], "questions": r["questions"]} for r in test]
        print("test accuracy by type:", accuracy_by_type(model, test_unlabeled))


if __name__ == "__main__":
    main()
