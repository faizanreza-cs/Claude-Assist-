"""Command line entry point: ``pharmapulse build-data | ask | eval``."""

from __future__ import annotations

import argparse
import sys

from pharmapulse.config import get_settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pharmapulse")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-data", help="generate the synthetic DuckDB database")
    b.add_argument("--path", default=str(get_settings().db_path))
    b.add_argument("--seed", type=int, default=42)
    a = sub.add_parser("ask", help="ask the agent a question")
    a.add_argument("question")
    a.add_argument("--provider", default=None)
    a.add_argument("--trace", action="store_true", help="print tool calls")
    sub.add_parser("eval", help="run the eval suite (see --help of the module)")
    args, rest = p.parse_known_args(argv)

    if args.cmd == "build-data":
        from pharmapulse.data.synth import SynthConfig, build_database

        path = build_database(args.path, SynthConfig(seed=args.seed))
        print(f"wrote {path}")
    elif args.cmd == "ask":
        from pharmapulse.llm.agent import ask

        res = ask(args.question, args.provider)
        if args.trace:
            for s in res.steps:
                print(f"-> {s.tool}({s.input}) [{s.latency_ms} ms]{' ERROR' if s.is_error else ''}")
        print(res.answer)
    elif args.cmd == "eval":
        from pharmapulse.evals.run_evals import main as eval_main

        return eval_main(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
