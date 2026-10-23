# Serving

`sysone serve` turns a saved model into an HTTP API with the System One request and response
shape of TypeSafe's Jev (`POST /v1/systemone`). Clients written for Jev, including the official
`typesafe-sdk`, work against it by changing only the base URL.

```bash
pip install "sysone[serve]"
sysone serve your-org/support-triage            # a Hub repo id, or a local folder from model.save()
```

```text
sysone 0.5.0 serving support-triage at http://127.0.0.1:8000/v1/systemone
```

## Asking it

```bash
curl http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "jev-latest",
  "state": "I was charged twice for March. Please refund the duplicate.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "team":   {"type": "choice", "instructions": "Which team should handle it?",
               "criteria": {"billing": "charges and refunds", "technical": "bugs and outages"}}
  }
}'
```

```json
{"id": "dec-3f9c0e5a7b1d4c2e8a6f0b9d",
 "model": "support-triage",
 "answers": {
   "refund": {"type": "noul", "noul": 0.97, "probabilities": {"false": 0.03, "true": 0.97},
              "answer_confidence": 0.97},
   "team":   {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.94, "technical": 0.06},
              "confidence": 0.88, "answer_confidence": 0.94}},
 "usage": {"input_tokens": 61, "output_tokens": 0}}
```

- `state` is text, a JSON object, or a list of text (such as conversation turns).
- `questions` take the same form as in {meth}`~sysone.SystemOneModel.predict`, and the answers
  are exactly what `predict` returns; see [Predictions](predictions.md).
- `model` is accepted and ignored: one server serves one model, so a client that always sends
  `jev-latest` still works. The response names the model that answered.

With the typesafe SDK:

```python
from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(api_key="unused-without-a-key", base_url="http://127.0.0.1:8000")
result = client.system_one(state="I was charged twice for March.",
                           questions={"refund": Noul(instructions="Is the customer asking for money back?")})
result.answers["refund"].noul
```

or set `TYPESAFE_BASE_URL=http://127.0.0.1:8000` for code that already uses it.

## Endpoints

| endpoint | what it does |
|---|---|
| `POST /v1/systemone` | answers the questions about the state |
| `GET /v1/models` | the served model, as `{"models": [{"name", "description", "release_date"}]}` |
| `GET /healthz` | 200 once the model is loaded; needs no API key, for load balancers |

Errors follow Jev's:

| status | when | body |
|---|---|---|
| 422 | the body or a question doesn't validate | `{"detail": [{"loc": ["body", "questions", "team", "choice", "criteria"], "msg": "...", "type": "..."}]}`, naming the failing field |
| 401 | an API key is required and missing or wrong | `{"detail": {"message": "..."}}` |
| 400 | `"stream": true` | decisions are not streamed |

Every response carries an `x-request-id` header equal to its `id`.

## Options

| option | default | meaning |
|---|---|---|
| `--host` | `127.0.0.1` | this machine only; `0.0.0.0` listens on every network interface |
| `--port` | `8000` | |
| `--name` | the folder or repo name | the model id the API reports |
| `--device` | the model's own (`auto`) | `cuda`, `mps` or `cpu` |
| `--api-key-env VAR` | none | require `Authorization: Bearer <key>`, with the key read from environment variable `VAR` |
| `--abstain T` | none | mark answers below `answer_confidence` `T` as abstained |
| `--abstain-file F` | none | per-type thresholds from a JSON file, such as what {meth}`~sysone.SystemOneModel.abstention_threshold` returns |
| `--max-batch` | `32` | the most requests answered together |
| `--batch-wait-ms` | `5` | how long a request waits for others to batch with |

## API keys

Without `--api-key-env`, the server answers anyone who can reach it, which suits local use. To
require a key:

```bash
export SYSONE_API_KEY="$(openssl rand -hex 24)"
sysone serve your-org/support-triage --host 0.0.0.0 --api-key-env SYSONE_API_KEY
```

The key is read from the environment, never from the command line, so it doesn't end up in your
shell history or in the process list.

## Abstention

`--abstain 0.8` marks every answer below 0.8 `answer_confidence` with `"abstained": true`; its
answer and probabilities are still returned. To use per-type thresholds chosen on held-out data:

```python
import json
thresholds = model.abstention_threshold(held_out_rows, target_accuracy=0.95)
json.dump(thresholds, open("thresholds.json", "w"))
```

```bash
sysone serve your-org/support-triage --abstain-file thresholds.json
```

A request can also send its own `"abstain"` (a number, or a dict by question type or id), which
replaces the server's for that request. This is a sysone addition; Jev clients don't send it.

## Batching and throughput

Requests that arrive together are answered in one forward pass: the first waits up to
`--batch-wait-ms` for others, up to `--max-batch` requests. A request on its own is delayed by at
most that wait. The model runs in a background thread, so the server keeps accepting requests
during a forward pass. One server uses one device; for more throughput, run several servers
(one per GPU) behind a load balancer.

## Deploying

- Put a reverse proxy (nginx, Caddy, a cloud load balancer) in front for HTTPS, and point its
  health check at `/healthz`.
- Run one server per model; the request's `model` field doesn't route between models.
- The first start downloads the model and its base model from the Hub; later starts use the
  local cache.

## In your own app

{func}`sysone.serve.create_app` returns the FastAPI app, to add routes or middleware, or to run it
with your own server setup:

```python
from sysone import SystemOneModel
from sysone.serve import create_app

app = create_app(SystemOneModel.load("your-org/support-triage"), api_key="...", abstain=0.8)
# uvicorn my_module:app --workers 1
```

Use one worker per model: each worker loads its own copy of the model.
