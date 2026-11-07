"""Compare eval runs and measure judge-human agreement.

  python report.py compare runs/a.jsonl runs/b.jsonl
  python report.py failures runs/a.jsonl
  python report.py export runs/a.jsonl labels.csv --n 20
  python report.py show runs/a.jsonl mt-reorder 3
  python report.py agreement labels.csv
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(passed, total, z=1.96):
    """95% interval. With small samples the width matters as much as the rate."""
    if total == 0:
        return 0.0, 0.0
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize(rows):
    """Pass rates over eligible verdicts only. Harness errors are counted, never scored."""
    scored = [r for r in rows if "verdicts" in r]
    criteria = defaultdict(lambda: [0, 0])
    sessions_passed = 0
    for row in scored:
        eligible = [v for v in row["verdicts"] if v["eligible"]]
        sessions_passed += all(v["passed"] for v in eligible)
        for v in eligible:
            criteria[v["criterion"]][0] += v["passed"]
            criteria[v["criterion"]][1] += 1
    malformed = sum(len(s.get("schema_errors", [])) > 0 for r in scored for s in r["trace"])
    return {
        "sessions": (sessions_passed, len(scored)),
        "criteria": dict(criteria),
        "harness_errors": len(rows) - len(scored),
        "malformed_tool_calls": malformed,
    }


def fmt(passed, total):
    if total == 0:
        return "n/a"
    low, high = wilson(passed, total)
    return f"{passed}/{total} {passed / total:4.0%} [{low:.0%}-{high:.0%}]"


def compare(paths):
    runs = [summarize(load(p)) for p in paths]
    names = sorted({c for r in runs for c in r["criteria"]})
    print(f"{'':28}" + "".join(f"{p[-30:]:>30}" for p in paths))
    print(f"{'session pass rate':28}" + "".join(f"{fmt(*r['sessions']):>30}" for r in runs))
    for name in names:
        print(f"{name:28}" + "".join(f"{fmt(*r['criteria'].get(name, (0, 0))):>30}" for r in runs))
    print(f"{'malformed tool calls':28}" + "".join(f"{r['malformed_tool_calls']:>30}" for r in runs))
    print(f"{'harness errors':28}" + "".join(f"{r['harness_errors']:>30}" for r in runs))


def failures(path):
    """Failures clustered by rubric criterion, most frequent first."""
    clusters = defaultdict(list)
    for row in load(path):
        for v in row.get("verdicts", []):
            if v["eligible"] and not v["passed"]:
                clusters[v["criterion"]].append((row["scenario"], row["trial"], v["rationale"]))
    for criterion, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{criterion}: {len(items)} failures")
        for scenario, trial, rationale in items:
            print(f"  {scenario} #{trial}: {rationale}")


def export(path, out, n, seed=0):
    """Sample verdicts for hand labeling. Hide the judge column while you label, or the labels are biased."""
    rows = [
        {"scenario": r["scenario"], "trial": r["trial"], "criterion": v["criterion"], "judge": int(v["passed"]), "human": ""}
        for r in load(path)
        for v in r.get("verdicts", [])
        if v["eligible"]
    ]
    random.Random(seed).shuffle(rows)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "trial", "criterion", "human", "judge"])
        writer.writeheader()
        writer.writerows(rows[:n])
    print(f"wrote {min(n, len(rows))} rows to {out}. Fill in human with 1 or 0. Hide the judge column while labeling.")


def show(path, scenario, trial):
    """Print one session's trace, so a human can read what the judge graded."""
    for row in load(path):
        if row["scenario"] == scenario and row["trial"] == trial:
            print(json.dumps(row.get("trace", row), indent=1))
            return
    raise SystemExit(f"No session {scenario} #{trial} in {path}")


def agreement_stats(pairs):
    """pairs: (judge, human). Returns raw agreement, Cohen's kappa, false positives, false negatives."""
    total = len(pairs)
    agree = sum(j == h for j, h in pairs) / total
    judge_yes = sum(j for j, _ in pairs) / total
    human_yes = sum(h for _, h in pairs) / total
    chance = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    kappa = 1.0 if chance == 1 else (agree - chance) / (1 - chance)
    return {
        "n": total,
        "agreement": agree,
        "kappa": kappa,
        "false_passes": sum(j == 1 and h == 0 for j, h in pairs),
        "false_fails": sum(j == 0 and h == 1 for j, h in pairs),
    }


def agreement(path):
    with open(path, newline="") as f:
        pairs = [(int(r["judge"]), int(r["human"])) for r in csv.DictReader(f) if r["human"].strip()]
    if not pairs:
        raise SystemExit("No human labels found. Fill in the human column first.")
    stats = agreement_stats(pairs)
    print(
        f"n={stats['n']} agreement={stats['agreement']:.0%} kappa={stats['kappa']:.2f} "
        f"false_passes={stats['false_passes']} false_fails={stats['false_fails']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("compare").add_argument("paths", nargs="+")
    sub.add_parser("failures").add_argument("path")
    export_parser = sub.add_parser("export")
    export_parser.add_argument("path")
    export_parser.add_argument("out")
    export_parser.add_argument("--n", type=int, default=20)
    show_parser = sub.add_parser("show")
    show_parser.add_argument("path")
    show_parser.add_argument("scenario")
    show_parser.add_argument("trial", type=int)
    sub.add_parser("agreement").add_argument("path")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.paths)
    elif args.command == "failures":
        failures(args.path)
    elif args.command == "export":
        export(args.path, args.out, args.n)
    elif args.command == "show":
        show(args.path, args.scenario, args.trial)
    else:
        agreement(args.path)
