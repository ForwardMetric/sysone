"""Serve a model over HTTP with the System One API (`POST /v1/systemone`), the request and response
shape of TypeSafe's Jev, so Jev clients work against it by changing only the base URL.

Requests are queued and answered in batches by one worker: it takes what is waiting, up to
`max_batch` requests, waiting at most `batch_wait_ms` for more once one has arrived. The model runs
in a worker thread, so the server keeps accepting requests during a forward pass.
"""

# No `from __future__ import annotations` here: FastAPI reads the endpoints' annotations at
# runtime, and `Request` is imported inside create_app so FastAPI stays an optional dependency.
import asyncio
import datetime
import hmac
import json
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .answers import check_abstain
from .questions import QUESTION_TYPES, parse_question

_FIELDS = ("criteria", "instructions", "type")


def _missing_extra() -> ImportError:
    return ImportError('serving needs FastAPI and uvicorn: pip install "sysone[serve]"')


def _error(loc: list, msg: str, kind: str = "value_error", value: Any = None) -> dict:
    error = {"loc": loc, "msg": msg, "type": kind}
    if value is not None:
        error["input"] = value
    return error


def validate_request(body: Any) -> list[dict]:
    """The request's problems, as FastAPI-style 422 `detail` entries naming each failing field."""
    if not isinstance(body, Mapping):
        return [_error(["body"], "the request body must be a JSON object", "type_error")]
    errors = []
    if "state" not in body:
        errors.append(_error(["body", "state"], "field required", "missing"))
    else:
        state = body["state"]
        if not (isinstance(state, (str, Mapping)) or
                (isinstance(state, list) and all(isinstance(s, str) for s in state))):
            errors.append(_error(["body", "state"], "state must be text, a JSON object or a list of text",
                                 "type_error"))
    questions = body.get("questions")
    if "questions" not in body:
        errors.append(_error(["body", "questions"], "field required", "missing"))
    elif not isinstance(questions, Mapping) or not questions:
        errors.append(_error(["body", "questions"], "questions must be a non-empty object of named questions",
                             "type_error"))
    else:
        for qid, question in questions.items():
            try:
                parse_question(qid, question)
            except ValueError as exc:
                # Jev's locations name the question's type before the field, as a tagged union does:
                # ["body", "questions", "urgency", "score", "criteria"]
                loc = ["body", "questions", qid]
                qtype = question.get("type") if isinstance(question, Mapping) else None
                field = next((f for f in _FIELDS if f in str(exc)), None)
                if field == "type" or qtype not in QUESTION_TYPES:
                    loc.append("type")
                else:
                    loc.append(qtype)
                    if field is not None:
                        loc.append(field)
                errors.append(_error(loc, str(exc), value=question))
    if "model" in body and not isinstance(body["model"], str):
        errors.append(_error(["body", "model"], "model must be a string", "type_error"))
    if body.get("abstain") is not None:
        try:
            check_abstain(body["abstain"])
        except ValueError as exc:
            errors.append(_error(["body", "abstain"], str(exc)))
    return errors


class Batcher:
    """Collects requests and answers them together, one batch at a time, off the event loop."""

    def __init__(self, model, max_batch: int = 32, batch_wait_ms: float = 5.0):
        self.model = model
        self.max_batch = max_batch
        self.batch_wait = batch_wait_ms / 1000
        self.queue: asyncio.Queue | None = None
        self.worker: asyncio.Task | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sysone-model")
        self.batch_sizes: list[int] = []  # for tests and monitoring

    async def answer(self, row: dict, abstain) -> dict:
        loop = asyncio.get_running_loop()
        if self.worker is None or self.worker.done() or self.worker.get_loop() is not loop:
            # the first request starts the worker on the server's event loop
            self.queue = asyncio.Queue()
            self.worker = loop.create_task(self._run())
        future = loop.create_future()
        await self.queue.put((row, abstain, future))
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.batch_wait
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                try:
                    batch.append(self.queue.get_nowait() if remaining <= 0 else
                                 await asyncio.wait_for(self.queue.get(), remaining))
                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                    break
            self.batch_sizes.append(len(batch))
            # requests with the same thresholds share one predict call
            groups: dict[str, list] = {}
            for item in batch:
                groups.setdefault(json.dumps(item[1], sort_keys=True, default=str), []).append(item)
            for items in groups.values():
                rows = [row for row, _, _ in items]
                try:
                    results = await loop.run_in_executor(self.executor, self._predict, rows, items[0][1])
                except Exception as exc:  # answer every waiting request rather than hang it
                    for _, _, future in items:
                        if not future.done():
                            future.set_exception(exc)
                    continue
                for (_, _, future), result in zip(items, results):
                    if not future.done():
                        future.set_result(result)

    def _predict(self, rows: list[dict], abstain) -> list[dict]:
        return self.model.predict(rows, abstain=abstain)


def create_app(model, name: str | None = None, api_key: str | None = None, abstain=None,
               max_batch: int = 32, batch_wait_ms: float = 5.0, description: str | None = None):
    """Build the FastAPI app that serves `model` with the System One API.

    Parameters
    ----------
    model : SystemOneModel
        A fitted or loaded model.
    name : str or None, default=None
        The model id the API reports. None uses the base model's name.
    api_key : str or None, default=None
        When set, requests must send ``Authorization: Bearer <api_key>``.
    abstain : float, dict or None, default=None
        Abstention thresholds applied to every request, as :meth:`SystemOneModel.predict` takes
        them; a request's own ``"abstain"`` replaces them.
    max_batch : int, default=32
        The most requests answered in one batch.
    batch_wait_ms : float, default=5.0
        How long the first request of a batch waits for others to join it.
    description : str or None, default=None
        The model's description in ``GET /v1/models``. None describes the base model and tuning.

    Returns
    -------
    fastapi.FastAPI
        Serve it with uvicorn, or test it with FastAPI's ``TestClient``.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise _missing_extra() from exc

    model._check_fitted()
    check_abstain(abstain)
    name = name or model.model_name
    description = description or "sysone %s on %s (%s tuning)" % (type(model).__name__, model.model_name,
                                                                  getattr(model, "tuning_", model.tuning))
    released = datetime.date.today().isoformat()
    batcher = Batcher(model, max_batch=max_batch, batch_wait_ms=batch_wait_ms)
    app = FastAPI(title="sysone", description="System One decisions from %s" % name, version="1")
    app.state.batcher = batcher

    def unauthorized(request: Request):
        if api_key is None:
            return None
        sent = request.headers.get("authorization", "")
        if hmac.compare_digest(sent.encode(), ("Bearer %s" % api_key).encode()):
            return None
        return JSONResponse({"detail": {"message": "Cannot authenticate: send Authorization: Bearer <key>"}},
                            status_code=401)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        denied = unauthorized(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"detail": [_error(["body"], "the request body is not valid JSON", "json_invalid")]},
                                status_code=422)
        if isinstance(body, Mapping) and body.get("stream"):
            return JSONResponse({"detail": "System One decisions are not streamed; remove \"stream\": true"},
                                status_code=400)
        errors = validate_request(body)
        if errors:
            return JSONResponse({"detail": errors}, status_code=422)
        request_abstain = body.get("abstain", abstain)
        result = await batcher.answer({"state": body["state"], "questions": body["questions"]}, request_abstain)
        decision_id = "dec-" + uuid.uuid4().hex[:24]
        return JSONResponse({"id": decision_id, "model": name, **result}, headers={"x-request-id": decision_id})

    @app.get("/v1/models")
    async def models(request: Request):
        denied = unauthorized(request)
        if denied is not None:
            return denied
        # the shape Jev's GET /v1/models returns, which the typesafe SDK reads
        return {"models": [{"name": name, "description": description, "release_date": released}]}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "model": name}

    return app
