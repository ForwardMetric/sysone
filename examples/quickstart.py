"""Quickstart: train a decision model on mixed support tickets, then answer new questions.

    uv run python examples/quickstart.py            # Qwen3-0.6B
    uv run python examples/quickstart.py encoder    # ModernBERT-base
"""

import random
import sys

from sysone import DecoderModel, EncoderModel

rng = random.Random(0)
TEAMS = {
    "sales": ["a quote", "pricing", "a discount", "a contract renewal"],
    "operations": ["a delayed shipment", "the warehouse", "a tracking number", "a late delivery"],
    "servicing": ["a refund", "a broken unit", "a repair", "returning the item"],
}
LEVELS = ["none", "minor", "major", "critical"]
LEVEL_WORDS = ["just a question", "a small issue", "a serious problem", "an emergency, everything is down"]

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle it?", "criteria": list(TEAMS)},
    "severity": {"type": "score", "instructions": "How severe is it?", "criteria": LEVELS},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}


def ticket(i):
    team = rng.choice(list(TEAMS))
    level = rng.randrange(len(LEVELS))
    topic = rng.choice(TEAMS[team])
    return {
        "state": "Ticket %d: Hi, this is about %s. It is %s." % (i, topic, LEVEL_WORDS[level]),
        "category": team,
        "questions": QUESTIONS,
        "expected": {"team": team, "severity": level, "refund": topic == "a refund"},
    }


def main():
    klass = EncoderModel if sys.argv[1:] == ["encoder"] else DecoderModel
    model = klass(tuning="head", batch_size=32)
    model.fit([ticket(i) for i in range(400)])
    print("validation:", model.report_["validation"])

    result = model.predict("Ticket 7: Our order is stuck in the warehouse and the line is down.", QUESTIONS)
    for qid, ans in result["answers"].items():
        print(qid, ans)

    model.save("quickstart-model")
    reloaded = klass.load("quickstart-model")
    assert reloaded.predict("Ticket 8: I would like a refund.", QUESTIONS)["answers"]["refund"]["type"] == "noul"


if __name__ == "__main__":
    main()
