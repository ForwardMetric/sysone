from pathlib import Path

import pytest

from sysone import DecoderModel, NotFittedError, SystemOneModel
from sysone.hub import model_card

from .conftest import QUESTIONS, make_rows

FAST = dict(tuning="head", epochs=1, batch_size=16, precision="fp32", device="cpu", verbose=0)


class FakeApi:
    """Records what push_to_hub sends instead of calling the Hub."""

    calls: list = []

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, repo_id, **kwargs):
        FakeApi.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, folder_path, repo_id, **kwargs):
        files = sorted(p.name for p in Path(folder_path).iterdir())
        readme = (Path(folder_path) / "README.md").read_text()
        FakeApi.calls.append(("upload_folder", repo_id, files, readme, kwargs))
        return type("Commit", (), {"commit_url": "https://huggingface.co/%s/commit/abc" % repo_id})()


@pytest.fixture
def fitted(tiny_models):
    return DecoderModel(tiny_models["qwen3"], **FAST).fit(make_rows(20))


def test_model_card_has_metadata_usage_and_training(fitted):
    card = model_card(fitted, "me/triage", license="apache-2.0")
    assert card.startswith("---\nlibrary_name: sysone\nbase_model: %s\n" % fitted.model_name)
    assert "license: apache-2.0" in card
    assert 'SystemOneModel.load("me/triage")' in card
    assert "sysone serve me/triage" in card
    assert "`DecoderModel`, `tuning=\"head\"`" in card
    assert "training and" in card and "Validation accuracy" in card
    assert "gated" not in card


def test_model_card_warns_about_gated_base_models(fitted):
    fitted.model_name = "meta-llama/Llama-3.2-1B"
    assert "The base model is gated" in model_card(fitted, "me/triage")


def test_push_to_hub_uploads_model_and_card_without_training_state(fitted, monkeypatch, tmp_path):
    import huggingface_hub

    FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    url = fitted.push_to_hub("me/triage", private=True)
    assert url == "https://huggingface.co/me/triage/commit/abc"
    (_, repo, kwargs), (_, _, files, readme, _) = FakeApi.calls
    assert repo == "me/triage" and kwargs["private"] is True and kwargs["exist_ok"] is True
    assert "README.md" in files and "config.json" in files and "head.safetensors" in files
    assert "training_state" not in files
    assert "library_name: sysone" in readme


def test_push_to_hub_needs_a_fitted_model(tiny_models):
    with pytest.raises(NotFittedError):
        SystemOneModel(tiny_models["qwen3"], **FAST).push_to_hub("me/triage")


def test_noul_answers_carry_probabilities(fitted):
    answer = fitted.predict("ticket: red minor refund", QUESTIONS)["answers"]["urgent"]
    assert answer["probabilities"]["true"] == answer["noul"]
    assert answer["probabilities"]["false"] + answer["probabilities"]["true"] == pytest.approx(1, abs=1e-3)
