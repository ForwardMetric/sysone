"""Prompt formats: the text of each piece of a question.

A model trained on one wording reads another as a different input, so the format is saved with
the model and reused at prediction time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .questions import Question


def _default_task_lines() -> dict:
    return {
        "choice": "Choose the best option.",
        "score": "Rate it on the given scale.",
        "noul": "Answer yes or no.",
    }


@dataclass
class PromptFormat:
    task_lines: dict = field(default_factory=_default_task_lines)
    question: str = "Question: {instructions}"
    choice_option: str = "{label}: {description}"
    score_option: str = "{index}: {level}"
    noul_labels: tuple = ("No", "Yes")

    def head(self, question: Question) -> str:
        """The task line and the question, as one piece of text."""
        return "%s\n%s" % (self.task_lines[question.type], self.question.format(instructions=question.instructions))

    def options(self, question: Question) -> list[str]:
        """The text of each option, in the question's option order."""
        if question.type == "choice":
            return [
                str(key) if text is None else self.choice_option.format(label=key, description=text)
                for key, text in zip(question.keys, question.texts)
            ]
        if question.type == "score":
            # The index must be in the text: in the parallel layout every option shares position
            # ids, so the text is the only place the order of the levels survives.
            return [self.score_option.format(index=i, level=text) for i, text in enumerate(question.texts)]
        return [
            label if text is None else "%s: %s" % (label, text)
            for label, text in zip(self.noul_labels, question.texts)
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["noul_labels"] = list(self.noul_labels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PromptFormat:
        d = dict(d)
        if "noul_labels" in d:
            d["noul_labels"] = tuple(d["noul_labels"])
        return cls(**d)
