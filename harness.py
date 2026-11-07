"""Mini eval harness: rubric -> transcript builder -> conversation simulator -> LLM-as-a-judge."""
import json, pathlib
from concurrent.futures import ThreadPoolExecutor
import anthropic
from agent import DATA, USAGE, Agent, HarnessError, ask, text

PROPS = {"rationale": {"type": "string"}, "eligible": {"type": "boolean"}, "passed": {"type": "boolean"}}
VERDICT = {"format": {"type": "json_schema", "schema": {"type": "object", "properties": PROPS,
           "required": list(PROPS), "additionalProperties": False}}}
SHOPPER = "You are a simulated user. Your plan: {}\nWrite only your next short message. When the plan is finished, write DONE."
JUDGE = "You are an LLM judge. Grade only from the evidence.\nEligible when: {applies_when}\nCriterion: {criterion}\nIf not eligible, set passed to false."

def strip(x):  # transcript builder: remove scaffolding that carries no quality signal
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if not k.startswith("_")}
    return [strip(v) for v in x] if isinstance(x, list) else x

def view(trace, evidence):  # each criterion sees only the span types it declares
    return json.dumps([strip(s) for s in trace if s["type"] in evidence], indent=1)

def simulate(client, scenario, args):  # simulated user drives the agent against frozen fixtures
    agent, msg = Agent(client, args.model, scenario["fixtures"], args.prompt, args.guard), scenario["startingPrompt"]
    for _ in range(scenario["max_turns"]):
        agent.send(msg)
        chat = [{"role": "user", "content": view(agent.trace, {"user", "assistant"})}]
        msg = text(ask(client, args.user_model, SHOPPER.format(scenario["conversationPlan"]), chat))
        if "DONE" in msg: break
    return agent.finish()

def judge(client, model, c, trace):  # one independent verdict per criterion
    chat = [{"role": "user", "content": view(trace, c["evidence"])}]
    return {"criterion": c["id"], **json.loads(text(ask(client, model, JUDGE.format(**c), chat, output_config=VERDICT)))}

def trial(client, args, rubric, scenario, n):
    row = {"scenario": scenario["evalId"], "trial": n, "model": args.model}
    try:
        row["trace"] = simulate(client, scenario, args)
        row["verdicts"] = [judge(client, args.judge, c, row["trace"]) for c in rubric + scenario.get("checklist", [])]
    except (anthropic.APIError, HarnessError, ValueError) as e:
        row["harness_error"] = repr(e)  # an infrastructure failure is not an agent failure
    return row

def run(client, args):
    rubric, scenarios = (json.loads((DATA / f).read_text()) for f in ("rubric.json", "scenarios.json"))
    jobs = [(s, n) for s in scenarios for n in range(args.trials)]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda j: trial(client, args, rubric, *j), jobs))
    pathlib.Path(args.out).parent.mkdir(exist_ok=True)
    pathlib.Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} sessions -> {args.out} | tokens {USAGE}")
