"""The agent under test: a small grocery agent whose tools read frozen fixtures.

Every step is appended to `trace` as a span. The harness never reaches into the
agent; it only reads the trace, the same way a production eval would.
"""
import json
import pathlib
import threading

DATA = pathlib.Path(__file__).parent / "data"
MAX_STEPS = 12  # model calls allowed per user message
USAGE = {"input": 0, "output": 0}
_usage_lock = threading.Lock()

TOOLS = [
    {
        "name": "get_reorder_items",
        "description": "Get the shopper's past grocery orders.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_catalog",
        "description": "Search the store catalog. Returns matching items with id, price, stock and allergens.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Plain text search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "set_cart_item",
        "description": "Set an item's quantity in the cart. Quantity 0 removes it. Returns the cart.",
        "input_schema": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}, "quantity": {"type": "integer"}},
            "required": ["item_id", "quantity"],
        },
    },
]
SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}
PY_TYPES = {"string": str, "integer": int}


class HarnessError(Exception):
    """The run itself failed. Reported separately and never scored as an agent failure."""


def ask(client, model, system, messages, **kwargs):
    response = client.messages.create(model=model, max_tokens=16000, system=system, messages=messages, **kwargs)
    with _usage_lock:
        USAGE["input"] += response.usage.input_tokens
        USAGE["output"] += response.usage.output_tokens
    if response.stop_reason in ("refusal", "max_tokens"):
        raise HarnessError(f"{model} stopped with {response.stop_reason}")
    return response


def text(response):
    return "".join(b.text for b in response.content if b.type == "text")


def schema_errors(name, args):
    """Compare a tool call to its declared schema, so malformed inputs show up in the trace."""
    schema = SCHEMAS.get(name)
    if schema is None:
        return [f"unknown tool {name}"]
    errors = [f"missing {key}" for key in schema["required"] if key not in args]
    for key, value in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            errors.append(f"unexpected {key}")
        elif not isinstance(value, PY_TYPES[spec["type"]]) or isinstance(value, bool):
            errors.append(f"{key} should be {spec['type']}, got {type(value).__name__}")
    return errors


def coerce(name, args):
    """Deterministic guard: repair the input shape instead of dropping the call."""
    fixed = dict(args)
    for key, spec in SCHEMAS.get(name, {}).get("properties", {}).items():
        value = fixed.get(key)
        if spec["type"] == "string" and isinstance(value, (dict, list)):
            parts = value.values() if isinstance(value, dict) else value
            flat = [p for part in parts for p in (part if isinstance(part, list) else [part])]
            fixed[key] = " ".join(str(p) for p in flat)
        elif spec["type"] == "integer" and isinstance(value, (str, float)):
            try:
                fixed[key] = int(float(value))
            except ValueError:
                pass
    return fixed


class FixtureTools:
    """Tools backed by recorded payloads, so every run sees the same world."""

    def __init__(self, packs):
        self.data = {}
        for pack in packs:
            self.data.update(json.loads((DATA / "fixtures" / f"{pack}.json").read_text()))
        self.cart = {}

    def call(self, name, args):
        if name == "get_reorder_items":
            return self.data["get_reorder_items"], False
        if name == "search_catalog":
            return self._search(args["query"]), False
        return self._set_cart_item(args["item_id"], args["quantity"])

    def _items(self):
        return self.data["search_catalog"]["items"]

    def _search(self, query):
        words = query.lower().split()
        scored = []
        for item in self._items():
            haystack = " ".join([item["name"], *item["tags"]]).lower()
            hits = sum(word in haystack for word in words)
            if hits:
                scored.append((-hits, item["id"], item))
        return {"_meta": self.data["search_catalog"]["_meta"], "items": [s[2] for s in sorted(scored, key=lambda s: s[:2])[:8]]}

    def _set_cart_item(self, item_id, quantity):
        item = next((i for i in self._items() if i["id"] == item_id), None)
        if item is None:
            return {"error": f"unknown item_id {item_id}"}, True
        if quantity < 0:
            return {"error": "quantity must be 0 or more"}, True
        if quantity > 0 and not item["in_stock"]:
            return {"error": f"{item['name']} is out of stock"}, True
        if quantity == 0:
            self.cart.pop(item_id, None)
        else:
            self.cart[item_id] = quantity
        return self.summary(), False

    def summary(self):
        lines = []
        for item_id, quantity in sorted(self.cart.items()):
            item = next(i for i in self._items() if i["id"] == item_id)
            lines.append({"id": item_id, "name": item["name"], "quantity": quantity, "price": item["price"]})
        return {"items": lines, "subtotal": round(sum(l["price"] * l["quantity"] for l in lines), 2)}


class Agent:
    def __init__(self, client, model, packs, prompt="system_prompt.txt", guard=False):
        self.client, self.model, self.guard = client, model, guard
        self.system = (DATA / prompt).read_text()
        self.tools = FixtureTools(packs)
        self.messages, self.trace = [], []

    def send(self, message):
        self.trace.append({"type": "user", "content": message})
        self.messages.append({"role": "user", "content": message})
        for _ in range(MAX_STEPS):
            response = ask(self.client, self.model, self.system, self.messages, tools=TOOLS)
            self.messages.append({"role": "assistant", "content": response.content})
            if reply := text(response):
                self.trace.append({"type": "assistant", "content": reply})
            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                return
            self.messages.append({"role": "user", "content": [self._run(c) for c in calls]})
        self.trace.append({"type": "step_limit", "content": f"stopped after {MAX_STEPS} model calls"})

    def _run(self, call):
        args = coerce(call.name, call.input) if self.guard else call.input
        errors = schema_errors(call.name, args)
        self.trace.append({"type": "tool_call", "name": call.name, "args": call.input, "schema_errors": errors})
        result, is_error = ({"error": "; ".join(errors)}, True) if errors else self.tools.call(call.name, args)
        self.trace.append({"type": "tool_result", "name": call.name, "content": result, "is_error": is_error})
        return {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result), "is_error": is_error}

    def finish(self):
        self.trace.append({"type": "final_cart", "content": self.tools.summary()})
        return self.trace
