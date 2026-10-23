import asyncio
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sysone import DecoderModel, NotFittedError  # noqa: E402
from sysone.cli import _parser  # noqa: E402
from sysone.serve import Batcher, create_app, validate_request  # noqa: E402

from .conftest import QUESTIONS, make_rows  # noqa: E402

FAST = dict(tuning="head", epochs=1, batch_size=16, precision="fp32", device="cpu", verbose=0)
STATE = "ticket: red minor refund"


@pytest.fixture(scope="module")
def model(tiny_models):
    return DecoderModel(tiny_models["qwen3"], **FAST).fit(make_rows(20))


@pytest.fixture
def client(model):
    with TestClient(create_app(model, name="tiny")) as c:
        yield c


def test_answers_match_predict_in_jev_shape(client, model):
    response = client.post("/v1/systemone", json={"model": "jev-latest", "state": STATE, "questions": QUESTIONS})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "tiny"  # the request's model is accepted and ignored
    assert body["id"].startswith("dec-") and response.headers["x-request-id"] == body["id"]
    expected = model.predict(STATE, QUESTIONS)
    assert body["answers"] == expected["answers"] and body["usage"] == expected["usage"]


def test_state_can_be_an_object_or_a_list_of_text(client, model):
    for state in ({"ticket": "red", "note": "refund"}, ["customer: refund please", "agent: sure"]):
        body = client.post("/v1/systemone", json={"state": state, "questions": QUESTIONS}).json()
        assert body["answers"] == model.predict([{"state": state, "questions": QUESTIONS}])[0]["answers"]


@pytest.mark.parametrize("body, loc", [
    ({"questions": QUESTIONS}, ["body", "state"]),
    ({"state": STATE}, ["body", "questions"]),
    ({"state": 3, "questions": QUESTIONS}, ["body", "state"]),
    ({"state": STATE, "questions": {}}, ["body", "questions"]),
    ({"state": STATE, "questions": {"refund": {"type": "maybe", "instructions": "x"}}},
     ["body", "questions", "refund", "type"]),
    ({"state": STATE, "questions": {"team": {"type": "choice", "instructions": "x", "criteria": ["a"]}}},
     ["body", "questions", "team", "choice"]),
    ({"state": STATE, "questions": {"team": {"type": "choice", "instructions": "x", "criteria": 5}}},
     ["body", "questions", "team", "choice", "criteria"]),
    ({"state": STATE, "questions": {"refund": {"type": "noul"}}},
     ["body", "questions", "refund", "noul", "instructions"]),
    ({"state": STATE, "questions": QUESTIONS, "abstain": 2}, ["body", "abstain"]),
])
def test_invalid_requests_name_the_failing_field(client, body, loc):
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    assert loc in [e["loc"] for e in response.json()["detail"]]


def test_body_must_be_a_json_object(client):
    assert client.post("/v1/systemone", content=b"not json").status_code == 422
    assert client.post("/v1/systemone", json=[1, 2]).json()["detail"][0]["loc"] == ["body"]


def test_streaming_is_refused(client):
    response = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS, "stream": True})
    assert response.status_code == 400 and "not streamed" in response.json()["detail"]


def test_api_key(model):
    with TestClient(create_app(model, api_key="s3cret")) as c:
        request = {"state": STATE, "questions": QUESTIONS}
        assert c.post("/v1/systemone", json=request).status_code == 401
        assert c.post("/v1/systemone", json=request, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.post("/v1/systemone", json=request, headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/healthz").status_code == 200  # health checks need no key


def test_abstention_from_the_server_and_the_request(model):
    request = {"state": STATE, "questions": QUESTIONS}
    with TestClient(create_app(model, abstain=1.0)) as c:
        answers = c.post("/v1/systemone", json=request).json()["answers"]
        assert all(a["abstained"] for a in answers.values())
        answers = c.post("/v1/systemone", json={**request, "abstain": 0.0}).json()["answers"]
        assert not any(a["abstained"] for a in answers.values())
    with TestClient(create_app(model)) as c:
        assert all("abstained" not in a for a in c.post("/v1/systemone", json=request).json()["answers"].values())


def test_models_and_health(client):
    [entry] = client.get("/v1/models").json()["models"]  # Jev's shape
    assert entry["name"] == "tiny" and "DecoderModel" in entry["description"]
    assert len(entry["release_date"]) == 10
    assert client.get("/healthz").json() == {"status": "ok", "model": "tiny"}


def test_concurrent_requests_are_batched_and_each_gets_its_own_answer(model):
    states = ["ticket %d: %s" % (i, color) for i, color in enumerate(["red", "green", "blue"] * 4)]
    batcher = Batcher(model, max_batch=32, batch_wait_ms=50)

    async def ask_all():
        return await asyncio.gather(*(batcher.answer({"state": s, "questions": QUESTIONS}, None) for s in states))

    results = asyncio.run(ask_all())
    assert batcher.batch_sizes == [len(states)]
    expected = model.predict([{"state": s, "questions": QUESTIONS} for s in states])
    assert results == expected


def test_batches_respect_max_batch_and_split_by_thresholds(model):
    batcher = Batcher(model, max_batch=4, batch_wait_ms=50)

    async def ask_all():
        return await asyncio.gather(*(batcher.answer({"state": "x %d" % i, "questions": QUESTIONS},
                                                     1.0 if i % 2 else None) for i in range(10)))

    results = asyncio.run(ask_all())
    assert max(batcher.batch_sizes) <= 4 and sum(batcher.batch_sizes) == 10
    assert all(a["abstained"] for a in results[1]["answers"].values())
    assert all("abstained" not in a for a in results[0]["answers"].values())


def test_model_errors_reach_the_request_instead_of_hanging(model, monkeypatch):
    batcher = Batcher(model, batch_wait_ms=1)
    monkeypatch.setattr(batcher, "_predict", lambda rows, abstain: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(batcher.answer({"state": STATE, "questions": QUESTIONS}, None))


def test_validate_request_accepts_a_good_request():
    assert validate_request({"model": "x", "state": STATE, "questions": QUESTIONS, "abstain": {"choice": 0.5}}) == []


def test_create_app_needs_a_fitted_model(tiny_models):
    with pytest.raises(NotFittedError):
        create_app(DecoderModel(tiny_models["qwen3"], **FAST))


def test_cli_arguments():
    args = _parser().parse_args(["serve", "me/model", "--port", "9000", "--abstain", "0.7", "--api-key-env", "KEY"])
    assert (args.model, args.port, args.abstain, args.api_key_env, args.host) == ("me/model", 9000, 0.7, "KEY",
                                                                                 "127.0.0.1")
    with pytest.raises(SystemExit):
        _parser().parse_args(["serve", "m", "--abstain", "0.5", "--abstain-file", "f.json"])


def test_cli_serve_loads_the_model_and_starts_uvicorn(model, tmp_path, monkeypatch):
    import uvicorn

    from sysone.cli import main

    folder = model.save(tmp_path / "triage-model")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"choice": {"threshold": 0.5}, "noul": None}))
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(app=app, **kwargs))
    monkeypatch.setenv("SYSONE_TEST_KEY", "k")
    main(["serve", str(folder), "--port", "9001", "--abstain-file", str(thresholds),
          "--api-key-env", "SYSONE_TEST_KEY"])
    assert started["port"] == 9001 and started["host"] == "127.0.0.1"
    with TestClient(started["app"]) as c:
        assert c.get("/v1/models").status_code == 401
        models = c.get("/v1/models", headers={"Authorization": "Bearer k"}).json()
        assert models["models"][0]["name"] == "triage-model"


def test_cli_refuses_an_unset_key_variable(model, tmp_path, monkeypatch):
    import uvicorn

    from sysone.cli import main

    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    monkeypatch.delenv("SYSONE_UNSET_KEY", raising=False)
    with pytest.raises(SystemExit, match="SYSONE_UNSET_KEY"):
        main(["serve", str(model.save(tmp_path / "m")), "--api-key-env", "SYSONE_UNSET_KEY"])
