# doordash-eval-harness

I Rebuilt DoorDash's Eval Harness in 50 Lines.

A mini eval harness for tool-calling agents, built from the patterns in DoorDash's post [Building Ask DoorDash (Part 3): Evaluation](https://careersatdoordash.com/blog/building-ask-doordash-part-three-evaluation/).

This is an independent teaching project. It isn't DoorDash's code and isn't affiliated with DoorDash. Everything here is my reading of a public blog post, scaled down.

**Status:** the plumbing is tested, but this repo contains no eval results yet. The tests run the real Anthropic SDK against a mocked HTTP transport, so they prove the code works and say nothing about any model.

## The four parts

The core is [`harness.py`](harness.py), under 50 lines, and a test enforces that. It doesn't include the agent under test, the reporting, or the data files.

| Part | Where | What it does |
|---|---|---|
| Rubric | [`data/rubric.json`](data/rubric.json), plus a `checklist` per scenario | Binary criteria. Each one has an eligibility condition and declares its evidence. |
| Transcript builder | `strip()` and `view()` | Strips scaffolding keys from spans, then gives each criterion only the span types it declares. |
| Conversation simulator | `simulate()` | A simulated user drives the agent through a scenario. Tools return fixtures, recorded payloads that keep every run on the same state. |
| LLM-as-a-judge | `judge()` | Scores each criterion independently, using structured output: `rationale`, `eligible`, `passed`. |

Calibration is separate: `report.py export` and `report.py agreement` measure judge-human agreement against your own labels.

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=your-key
```

## Run it

Start with a one-trial smoke test:

```bash
uv run cli.py --trials 1 --out runs/smoke.jsonl
```

Compare two base models, 8 trials per scenario. Keep the judge and simulated user fixed, or you're measuring them instead of the agent:

```bash
uv run cli.py --model claude-opus-5-5 --out runs/model_a.jsonl
uv run cli.py --model claude-haiku-5-5 --out runs/model_b.jsonl
uv run report.py compare runs/model_a.jsonl runs/model_b.jsonl
```

`compare` prints pass rates with 95% Wilson intervals. Trials of one scenario aren't independent samples, so treat those intervals as a lower bound on the uncertainty.

Read the failures, clustered by rubric criterion, then open a session:

```bash
uv run report.py failures runs/model_b.jsonl
uv run report.py show runs/model_b.jsonl mt-reorder 3
```

Test a fix. Write a new system prompt into `data/` and load it with `--prompt`. `--guard` turns on the deterministic guard, which coerces malformed tool inputs instead of rejecting them:

```bash
uv run cli.py --model claude-haiku-5-5 --prompt system_prompt_v2.txt --guard --out runs/model_b_fixed.jsonl
```

## Calibrate the judge

Don't trust a score until you've checked the judge:

```bash
uv run report.py export runs/model_a.jsonl labels.csv --n 20
```

Fill in the `human` column with 1 or 0. Use `report.py show` to read each session, and hide the `judge` column while you label. Then:

```bash
uv run report.py agreement labels.csv
```

This prints raw agreement, Cohen's kappa, and the counts of false passes and false fails.

## Design choices

- **Harness errors aren't agent failures.** API errors, refusals, truncated output and invalid judge JSON are counted separately and never scored.
- **Malformed tool calls are visible.** Every call is checked against its declared schema and the result is stored in the trace. Tools don't use `strict: true`, on purpose, so schema violations can appear. With `--guard` on, the trace keeps the original arguments.
- **No fallback models.** A silent fallback would change the model under test or the judge in the middle of a run. A refusal is recorded as a harness error.
- **Ineligible criteria don't count.** They're left out of pass rates entirely.

## Tests

```bash
uv run pytest
```

## How this differs from DoorDash's harness

This is a small subset. It has one agent instead of a multi-agent system, a fixture-backed catalog instead of production services, and JSONL files instead of OpenTelemetry traces in ClickHouse. There's no online eval path, and judge calibration is a manual agreement check, not GEPA prompt optimization.
