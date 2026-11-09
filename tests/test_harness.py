"""Plumbing tests. These use the real OpenAI SDK over a mocked HTTP transport.

They prove the harness builds valid requests, parses responses and scores
correctly. They say nothing about how any real model performs.
"""
import json
import pathlib
from types import SimpleNamespace

import openai
import httpx2 as httpx
import pytest

import agent as agent_mod
import harness
import report
from agent import Agent, FixtureTools, HarnessError, coerce, schema_errors
from cli import parse

ROOT = pathlib.Path(__file__).parent.parent


def completion(msg, finish_reason="stop"):
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", **msg}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def call_of(name, tool_input, call_id):
    arguments = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def say(words, finish_reason="stop"):
    return completion({"content": words}, finish_reason)


def refuse():
    return completion({"content": None, "refusal": "I can't help with that."})


def calls(*tool_calls):
    return completion({"content": None, "tool_calls": list(tool_calls)}, "tool_calls")


def tool(name, tool_input, call_id="call_1"):
    return calls(call_of(name, tool_input, call_id))


def verdict(passed=True, eligible=True):
    return say(json.dumps({"rationale": "because", "eligible": eligible, "passed": passed}))


def client_for(responder):
    """A real SDK client whose transport answers from `responder(request_body)`."""
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=responder(body))

    client = openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle)), max_retries=0)
    return client, requests


def scripted(*responses):
    queue = list(responses)
    return client_for(lambda body: queue.pop(0))


# --- Fixtures and tools ---

def test_search_returns_recorded_items_in_stable_order():
    tools = FixtureTools(["catalog_v1"])
    first, _ = tools.call("search_catalog", {"query": "taco beans"})
    second, _ = tools.call("search_catalog", {"query": "taco beans"})
    assert first == second
    assert first["items"][0]["id"] in {"black-beans", "pinto-beans"}


def test_out_of_stock_item_cannot_enter_cart():
    tools = FixtureTools(["catalog_v1"])
    result, is_error = tools.call("set_cart_item", {"item_id": "milk-oat", "quantity": 1})
    assert is_error and "out of stock" in result["error"]
    assert tools.summary()["items"] == []


def test_cart_quantity_change_removal_and_subtotal():
    tools = FixtureTools(["catalog_v1"])
    tools.call("set_cart_item", {"item_id": "avocado", "quantity": 2})
    tools.call("set_cart_item", {"item_id": "bread-wheat", "quantity": 1})
    tools.call("set_cart_item", {"item_id": "avocado", "quantity": 3})
    tools.call("set_cart_item", {"item_id": "bread-wheat", "quantity": 0})
    assert tools.summary() == {
        "items": [{"id": "avocado", "name": "Hass Avocado", "quantity": 3, "price": 1.49}],
        "subtotal": 4.47,
    }


def test_reorder_fixture_ids_exist_in_catalog():
    tools = FixtureTools(["catalog_v1", "reorder_history_v1"])
    ids = {i["id"] for i in tools._items()}
    ordered = {i["id"] for o in tools.data["get_reorder_items"]["orders"] for i in o["items"]}
    assert ordered <= ids


# --- Schema errors and the deterministic guard ---

def test_schema_errors_flag_wrong_type_missing_and_unknown():
    assert schema_errors("search_catalog", {"query": {"dishes": ["tacos"]}}) == ["query should be string, got dict"]
    assert schema_errors("set_cart_item", {"item_id": "eggs"}) == ["missing quantity"]
    assert schema_errors("search_catalog", {"query": "x", "limit": 3}) == ["unexpected limit"]
    assert schema_errors("nope", {}) == ["unknown tool nope"]
    assert schema_errors("set_cart_item", {"item_id": "eggs", "quantity": True}) == ["quantity should be integer, got bool"]


def test_guard_coerces_object_query_and_string_quantity():
    assert coerce("search_catalog", {"query": {"dishes": ["tacos", "salsa"]}}) == {"query": "tacos salsa"}
    assert coerce("set_cart_item", {"item_id": "eggs", "quantity": "2"})["quantity"] == 2
    assert coerce("set_cart_item", {"item_id": "eggs", "quantity": "two"})["quantity"] == "two"


def test_malformed_call_without_guard_returns_tool_error_and_is_traced():
    client, requests = scripted(tool("search_catalog", {"query": {"dishes": ["tacos"]}}), say("Sorry, no luck."))
    a = Agent(client, "m", ["catalog_v1"])
    a.send("tacos please")
    call = next(s for s in a.trace if s["type"] == "tool_call")
    result = next(s for s in a.trace if s["type"] == "tool_result")
    assert call["schema_errors"] and result["is_error"]
    assert json.loads(requests[1]["messages"][-1]["content"]) == {"error": "query should be string, got dict"}


def test_malformed_call_with_guard_runs_but_keeps_the_original_args_in_the_trace():
    client, _ = scripted(tool("search_catalog", {"query": {"dishes": ["tacos"]}}), say("Found some."))
    a = Agent(client, "m", ["catalog_v1"], guard=True)
    a.send("tacos please")
    call = next(s for s in a.trace if s["type"] == "tool_call")
    result = next(s for s in a.trace if s["type"] == "tool_result")
    assert call["args"] == {"query": {"dishes": ["tacos"]}}
    assert call["schema_errors"] == [] and not result["is_error"]


# --- Agent loop ---

def test_agent_replays_tool_use_and_matching_tool_result():
    client, requests = scripted(tool("search_catalog", {"query": "lime"}, "toolu_9"), say("Limes are $0.59."))
    a = Agent(client, "m", ["catalog_v1"])
    a.send("limes?")
    second = requests[1]
    assert second["messages"][0]["role"] == "system" and second["messages"][0]["content"].startswith("You are a grocery")
    assert second["messages"][2]["tool_calls"][0]["id"] == "toolu_9"
    assert second["messages"][3]["role"] == "tool" and second["messages"][3]["tool_call_id"] == "toolu_9"
    assert [t["function"]["name"] for t in second["tools"]] == ["get_reorder_items", "search_catalog", "set_cart_item"]


def test_agent_runs_parallel_tool_calls_and_returns_results_in_one_message():
    both = calls(call_of("search_catalog", {"query": "lime"}, "a"), call_of("search_catalog", {"query": "salsa"}, "b"))
    client, requests = scripted(both, say("Done."))
    Agent(client, "m", ["catalog_v1"]).send("hi")
    assert [m["tool_call_id"] for m in requests[1]["messages"][-2:]] == ["a", "b"]


def test_non_json_tool_arguments_become_a_traced_tool_error():
    client, _ = scripted(tool("search_catalog", "{not json"), say("Sorry."))
    a = Agent(client, "m", ["catalog_v1"])
    a.send("limes?")
    call = next(s for s in a.trace if s["type"] == "tool_call")
    assert call["schema_errors"] == ["arguments are not a JSON object"]


def test_step_limit_is_recorded_as_a_span():
    client, _ = client_for(lambda body: tool("search_catalog", {"query": "lime"}))
    a = Agent(client, "m", ["catalog_v1"])
    a.send("loop forever")
    assert a.trace[-1]["type"] == "step_limit"


@pytest.mark.parametrize("response", [refuse(), say("", "length"), say("", "content_filter")])
def test_truncated_or_refused_response_is_a_harness_error(response):
    client, _ = scripted(response)
    with pytest.raises(HarnessError):
        Agent(client, "m", ["catalog_v1"]).send("hi")


# --- Transcript builder ---

TRACE = [
    {"type": "user", "content": "hi"},
    {"type": "assistant", "content": "hello"},
    {"type": "tool_result", "name": "x", "content": {"_meta": {"schema_version": "1"}, "items": [{"id": "a", "_rank": 1}]}},
]


def test_view_keeps_only_declared_spans():
    spans = json.loads(harness.view(TRACE, ["assistant"]))
    assert spans == [{"type": "assistant", "content": "hello"}]


def test_view_strips_scaffolding_at_every_depth():
    spans = json.loads(harness.view(TRACE, ["tool_result"]))
    assert spans[0]["content"] == {"items": [{"id": "a"}]}


def test_every_rubric_evidence_type_is_produced_by_the_agent():
    known = {"user", "assistant", "tool_call", "tool_result", "final_cart"}
    rubric = json.loads((ROOT / "data/rubric.json").read_text())
    scenarios = json.loads((ROOT / "data/scenarios.json").read_text())
    criteria = rubric + [c for s in scenarios for c in s["checklist"]]
    assert all(set(c["evidence"]) <= known for c in criteria)
    assert all({"id", "applies_when", "criterion", "evidence"} <= set(c) for c in criteria)


# --- Judge ---

def test_judge_sends_structured_output_and_only_the_evidence_slice():
    client, requests = scripted(verdict(passed=False))
    c = {"id": "narration", "applies_when": "Always.", "criterion": "No tool names.", "evidence": ["assistant"]}
    result = harness.judge(client, "judge-model", c, TRACE)
    assert result == {"criterion": "narration", "rationale": "because", "eligible": True, "passed": False}
    body = requests[0]
    assert body["response_format"]["type"] == "json_schema"
    assert "tool_result" not in body["messages"][1]["content"]
    assert "No tool names." in body["messages"][0]["content"]


# --- Full run ---

def args_for(tmp_path, **overrides):
    return parse(["--trials", "1", "--workers", "1", "--out", str(tmp_path / "r.jsonl")] + overrides.get("argv", []))


def fake_world(body):
    """Answers as agent, simulated user or judge depending on the request."""
    if "response_format" in body:
        return verdict()
    if "tools" in body:
        return say("Happy to help.")
    return say("DONE")


def test_full_run_writes_one_row_per_scenario_trial(tmp_path):
    client, _ = client_for(fake_world)
    harness.run(client, args_for(tmp_path))
    rows = report.load(tmp_path / "r.jsonl")
    assert len(rows) == 6
    assert all(r["trace"][-1]["type"] == "final_cart" for r in rows)
    assert all(len(r["verdicts"]) >= 4 for r in rows)


def test_api_failure_becomes_a_harness_error_row_not_an_agent_failure(tmp_path):
    def handle(request):
        return httpx.Response(500, json={"type": "error", "error": {"type": "server_error", "message": "boom"}})

    client = openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle)), max_retries=0)
    harness.run(client, args_for(tmp_path))
    summary = report.summarize(report.load(tmp_path / "r.jsonl"))
    assert summary["harness_errors"] == 6
    assert summary["sessions"] == (0, 0)


def test_invalid_judge_json_is_a_harness_error(tmp_path):
    def world(body):
        return say("not json") if "response_format" in body else fake_world(body)

    client, _ = client_for(world)
    harness.run(client, args_for(tmp_path))
    assert report.summarize(report.load(tmp_path / "r.jsonl"))["harness_errors"] == 6


# --- Reporting ---

def row(*verdicts, trace=()):
    return {"scenario": "s", "trial": 0, "trace": list(trace), "verdicts": list(verdicts)}


def v(criterion, passed, eligible=True):
    return {"criterion": criterion, "passed": passed, "eligible": eligible, "rationale": "r"}


def test_ineligible_criteria_do_not_count_for_or_against():
    summary = report.summarize([row(v("a", True), v("b", False, eligible=False))])
    assert summary["sessions"] == (1, 1)
    assert summary["criteria"] == {"a": [1, 1]}


def test_one_failed_eligible_criterion_fails_the_session():
    assert report.summarize([row(v("a", True), v("b", False))])["sessions"] == (0, 1)


def test_malformed_tool_calls_are_counted():
    spans = [{"type": "tool_call", "schema_errors": ["bad"]}, {"type": "tool_call", "schema_errors": []}]
    assert report.summarize([row(trace=spans)])["malformed_tool_calls"] == 1


def test_wilson_interval_is_wide_for_small_samples():
    low, high = report.wilson(8, 10)
    assert low == pytest.approx(0.490, abs=0.01) and high == pytest.approx(0.943, abs=0.01)
    assert report.wilson(0, 0) == (0.0, 0.0)


def test_agreement_stats():
    stats = report.agreement_stats([(1, 1), (1, 1), (0, 0), (1, 0)])
    assert stats["agreement"] == 0.75
    assert stats["kappa"] == pytest.approx(0.5)
    assert stats["false_passes"] == 1 and stats["false_fails"] == 0


def test_export_then_agreement_round_trip(tmp_path, capsys):
    run = tmp_path / "r.jsonl"
    run.write_text(json.dumps(row(v("a", True), v("b", False))) + "\n")
    labels = tmp_path / "l.csv"
    report.export(run, labels, 10)
    text = labels.read_text().replace("\r\n", "\n").splitlines()
    assert text[0] == "scenario,trial,criterion,human,judge"
    filled = [text[0]] + [line.replace(",,", ",1,") for line in text[1:]]
    labels.write_text("\n".join(filled) + "\n")
    report.agreement(labels)
    assert "n=2 agreement=50%" in capsys.readouterr().out


def test_show_prints_the_trace_and_rejects_unknown_sessions(tmp_path, capsys):
    run = tmp_path / "r.jsonl"
    run.write_text(json.dumps(row(trace=[{"type": "user", "content": "hello there"}])) + "\n")
    report.show(run, "s", 0)
    assert "hello there" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        report.show(run, "s", 1)


# --- The title's claim ---

def test_core_is_50_lines_or_fewer():
    assert len((ROOT / "harness.py").read_text().splitlines()) <= 50


def test_tool_schemas_match_python_type_table():
    types = {p["type"] for s in agent_mod.SCHEMAS.values() for p in s["properties"].values()}
    assert types <= set(agent_mod.PY_TYPES)
