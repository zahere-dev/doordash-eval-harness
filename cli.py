"""Command line entry point for the 50-line core in harness.py."""
import argparse

import anthropic

from harness import run

DEFAULT_MODEL = "claude-opus-5-5"


def parse(argv=None):
    p = argparse.ArgumentParser(description="Run the mini eval harness.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="base model under test")
    p.add_argument("--judge", default=DEFAULT_MODEL, help="LLM judge model; keep it fixed across runs")
    p.add_argument("--user-model", default=DEFAULT_MODEL, help="simulated user model; keep it fixed across runs")
    p.add_argument("--trials", type=int, default=8, help="trials per scenario")
    p.add_argument("--workers", type=int, default=4, help="sessions run in parallel")
    p.add_argument("--prompt", default="system_prompt.txt", help="system prompt file inside data/")
    p.add_argument("--guard", action="store_true", help="coerce malformed tool inputs instead of rejecting them")
    p.add_argument("--out", default="runs/run.jsonl", help="results file")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(anthropic.Anthropic(), parse())
