"""The `sysone` command."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sysone", description="sysone: calibrated system one decision models")
    parser.add_argument("--version", action="version", version="sysone %s" % __version__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser(
        "serve", help="serve a model with the System One API (POST /v1/systemone)",
        description="Serve a saved sysone model over HTTP with the System One API, the request and "
                    "response shape of TypeSafe's Jev: Jev clients work against it by changing only the "
                    "base URL.")
    serve.add_argument("model", help="a saved model: a local folder or a Hugging Face Hub repo id")
    serve.add_argument("--host", default="127.0.0.1",
                       help="address to listen on (default: 127.0.0.1, this machine only; 0.0.0.0 for all)")
    serve.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    serve.add_argument("--name", help="the model id the API reports (default: the model's folder or repo name)")
    serve.add_argument("--device", default=None, help="cuda, mps or cpu (default: the model's setting, auto)")
    serve.add_argument("--api-key-env", metavar="VAR",
                       help="require Authorization: Bearer <key>, with the key read from environment variable VAR")
    thresholds = serve.add_mutually_exclusive_group()
    thresholds.add_argument("--abstain", type=float, metavar="T",
                            help="mark answers below this answer_confidence as abstained")
    thresholds.add_argument("--abstain-file", metavar="FILE",
                            help="a JSON file of thresholds, such as what abstention_threshold() returns")
    serve.add_argument("--max-batch", type=int, default=32, help="most requests answered together (default: 32)")
    serve.add_argument("--batch-wait-ms", type=float, default=5.0,
                       help="how long a request waits for others to batch with (default: 5)")
    return parser


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        sys.exit('sysone serve needs FastAPI and uvicorn: pip install "sysone[serve]"')
    from .base import SystemOneModel
    from .serve import create_app

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            sys.exit("--api-key-env %s: that environment variable is not set" % args.api_key_env)
    abstain = args.abstain
    if args.abstain_file:
        abstain = json.loads(Path(args.abstain_file).read_text(encoding="utf-8"))
    overrides = {"device": args.device} if args.device else {}
    model = SystemOneModel.load(args.model, **overrides)
    if args.name:
        name = args.name
    elif Path(args.model).exists():  # a local folder: its name
        name = Path(args.model).resolve().name
    else:  # a Hub repo id
        name = args.model
    app = create_app(model, name=name, api_key=api_key, abstain=abstain, max_batch=args.max_batch,
                     batch_wait_ms=args.batch_wait_ms)
    print("sysone %s serving %s at http://%s:%d/v1/systemone%s"
          % (__version__, name, args.host, args.port, " (API key required)" if api_key else ""), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        _serve(args)


if __name__ == "__main__":
    main()
