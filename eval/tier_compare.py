"""4-tier comparison scorer.

Scores tiers 1-4 against the same 63-ticket benchmark (adversarial
T064-T072 excluded) using a single blind LLM judge, and reports whether
tier 4 (the full agent) is actually ahead of tiers 1-3 (prose baselines),
using a strict min/max separation rule so no tier "wins" by a coin-flip
sized difference.

Tiers 1, 2, 3 produce prose (``result["baseline"]["final_answer"]``).
Tier 4 produces a TaskState (``result["state"]["hypothesis"]``). The judge
never sees which tier produced an answer -- see ``candidate_answer`` and
the ``judge_*`` wrappers below.

Run under the project venv:

    venv/bin/python -m eval.tier_compare --root eval/results/tier_compare --live

Costs real money (one judge call per ticket x tier x repeat, roughly
doubled for the 10 ambiguous tickets which are also abstention-judged) --
refuses to run without --live. Verdicts are cached on disk
(``<root>/judge_cache.json``) so a rerun over the same inputs is free;
pass ``--no-cache`` to bypass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import llm_judge, metrics, report

TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"

DEFAULT_ROOT = REPO_ROOT / "eval" / "results" / "tier_compare"
DEFAULT_OUT = REPO_ROOT / "eval" / "results" / "tier_compare.json"

CATEGORIES = ["easy", "multi_step", "tool_heavy", "rag_heavy"]

TIER4_ONLY_METRICS = [
    "task_success_status_only",
    "task_success_strict_lexical",
    "correct_escalation",
    "tool_selection_accuracy",
    "unnecessary_tool_calls",
    "parameter_validity",
    "state_consistency",
    "write_gate_appended_correctly",
    "citation_presence",
    "retrieval_recall_at_3_observed",
]

Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Ticket loading
# ---------------------------------------------------------------------------


def load_tickets() -> dict[str, dict]:
    with open(TICKETS_PATH, encoding="utf-8") as f:
        tickets = json.load(f)
    return {t["id"]: t for t in tickets}


# ---------------------------------------------------------------------------
# Answer extraction -- the single, non-fallback function
# ---------------------------------------------------------------------------


def candidate_answer(result: dict) -> str | None:
    """Extracts the free-text/prose answer to be judged for one result
    record. Never falls back between shapes: tiers 1-3 only ever read
    ``baseline.final_answer``; tier 4 only ever reads
    ``state.hypothesis``.
    """
    tier = result.get("tier")
    if tier in (1, 2, 3):
        baseline = result.get("baseline")
        if not baseline:
            return None
        return baseline.get("final_answer")
    if tier == 4:
        state = result.get("state")
        if not state:
            return None
        return state.get("hypothesis")
    raise ValueError(f"unknown tier: {tier!r}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_results(root: Path, tiers: list[int] | None = None) -> list[dict]:
    """Globs ``root/tier{N}/repeat{K}/*.json`` (repeat subdirectories are
    optional -- if absent, the repeat number is taken from the result's
    own ``"repeat"`` field, defaulting to 0). Returns a flat list of
    ``{"tier": N, "repeat": K, "ticket_id": ..., "result": {...}}``.
    """
    records = []
    for tier_dir in sorted(root.glob("tier*")):
        if not tier_dir.is_dir():
            continue
        m = re.match(r"tier(\d+)$", tier_dir.name)
        if not m:
            continue
        tier = int(m.group(1))
        if tiers is not None and tier not in tiers:
            continue
        for path in sorted(tier_dir.rglob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            rel = path.relative_to(tier_dir)
            repeat = None
            if len(rel.parts) > 1:
                rm = re.match(r"repeat(\d+)$", rel.parts[0])
                if rm:
                    repeat = int(rm.group(1))
            if repeat is None:
                repeat = result.get("repeat", 0)
            ticket_id = result.get("ticket_id") or path.stem
            records.append({"tier": tier, "repeat": repeat, "ticket_id": ticket_id, "result": result})
    return records


# ---------------------------------------------------------------------------
# Wilson interval (no scipy)
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    p = successes / trials
    denom = 1 + (z * z) / trials
    center = p + (z * z) / (2 * trials)
    margin = z * math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


# ---------------------------------------------------------------------------
# Cell aggregation across repeats
# ---------------------------------------------------------------------------


def aggregate_cell(per_repeat_success_trials: list[tuple[int, int]]) -> dict:
    """``per_repeat_success_trials`` is one (n_success, n_trials) pair per
    repeat. Returns n_repeats, the per-repeat rate list (None for a repeat
    with zero trials), median/min/max over the repeats that have data, and
    a pooled Wilson 95% interval over the SUMMED successes/trials.
    """
    per_repeat_rates = [(s / t if t else None) for s, t in per_repeat_success_trials]
    clean = [r for r in per_repeat_rates if r is not None]
    total_s = sum(s for s, _t in per_repeat_success_trials)
    total_t = sum(t for _s, t in per_repeat_success_trials)
    wilson_lo, wilson_hi = wilson_interval(total_s, total_t)
    return {
        "n_repeats": len(per_repeat_success_trials),
        "per_repeat": per_repeat_rates,
        "median": statistics.median(clean) if clean else None,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "n_success_total": total_s,
        "n_trials_total": total_t,
        "wilson_lo": wilson_lo,
        "wilson_hi": wilson_hi,
    }


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def compare_to_tier4(cell_tier4: dict, cell_other: dict) -> dict:
    t4_min, t4_max = cell_tier4.get("min"), cell_tier4.get("max")
    o_min, o_max = cell_other.get("min"), cell_other.get("max")

    if t4_min is None or t4_max is None or o_min is None or o_max is None:
        return {
            "verdict": "inconclusive",
            "detail": "insufficient data to compare (one or both cells have no repeats with trials)",
        }

    if t4_min > o_max:
        return {
            "verdict": "tier4_ahead",
            "detail": (
                f"tier4 range [{t4_min:.3f}, {t4_max:.3f}] is entirely above "
                f"the other tier's range [{o_min:.3f}, {o_max:.3f}]"
            ),
        }
    if o_min > t4_max:
        return {
            "verdict": "other_ahead",
            "detail": (
                f"the other tier's range [{o_min:.3f}, {o_max:.3f}] is entirely above "
                f"tier4's range [{t4_min:.3f}, {t4_max:.3f}]"
            ),
        }
    return {
        "verdict": "inconclusive",
        "detail": (
            f"overlapping ranges: tier4 [{t4_min:.3f}, {t4_max:.3f}] vs "
            f"other [{o_min:.3f}, {o_max:.3f}]; separating them needs more repeats"
        ),
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_key(prompt_variant: str, ticket_id: str, candidate: str | None) -> str:
    h = hashlib.sha256()
    h.update(prompt_variant.encode("utf-8"))
    h.update(b"|")
    h.update(ticket_id.encode("utf-8"))
    h.update(b"|")
    h.update((candidate or "").encode("utf-8"))
    return h.hexdigest()


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


class JudgeStats:
    def __init__(self):
        self.n_judge_calls_made = 0
        self.n_cache_hits = 0


# ---------------------------------------------------------------------------
# Blind judging wrappers
# ---------------------------------------------------------------------------


def judge_semantic(
    ticket_id: str,
    ticket_text: str,
    category: str,
    candidate: str,
    gold_root_cause: str,
    cache: dict,
    stats: JudgeStats,
    no_cache: bool = False,
) -> tuple[bool | None, str | None]:
    """Blind clause-(a)-only semantic-match judging, with disk caching.

    Uses eval.llm_judge.judge_semantic_match_once (referenced via the
    module so tests can monkeypatch it), which never receives a tier
    marker -- only ticket text, category, the candidate answer, and the
    gold slug.
    """
    key = cache_key(llm_judge.PROMPT_VARIANT_3A, ticket_id, candidate)
    if not no_cache and key in cache:
        stats.n_cache_hits += 1
        entry = cache[key]
        return entry.get("verdict"), entry.get("reasoning")

    verdict, reasoning, error = llm_judge.judge_semantic_match_once(
        ticket_text, category, candidate, gold_root_cause
    )
    stats.n_judge_calls_made += 1
    cache[key] = {"verdict": verdict, "reasoning": reasoning, "error": error}
    return verdict, reasoning


def judge_abstention(
    ticket_id: str,
    ticket_text: str,
    candidate: str,
    cache: dict,
    stats: JudgeStats,
    no_cache: bool = False,
) -> tuple[bool | None, str | None]:
    """Blind abstention judging, with disk caching. Judges the prose
    candidate answer directly -- for tier 4 this is state.hypothesis, NOT
    state.status == "escalated". Using status would be reading the
    system's own self-report of what it did, not judging what the
    hypothesis text actually says; the fairness rule for this comparison
    is that every tier's answer is judged the same way.
    """
    key = cache_key(llm_judge.PROMPT_VARIANT_ABSTENTION, ticket_id, candidate)
    if not no_cache and key in cache:
        stats.n_cache_hits += 1
        entry = cache[key]
        return entry.get("verdict"), entry.get("reasoning")

    verdict, reasoning, error = llm_judge.judge_abstention_once(ticket_text, candidate)
    stats.n_judge_calls_made += 1
    cache[key] = {"verdict": verdict, "reasoning": reasoning, "error": error}
    return verdict, reasoning


# Deliberately phrase-based, not the bare substring "tier" -- a legitimate
# candidate answer can contain that word (e.g. "the tier of service was
# degraded") without leaking which tier produced it, so rejecting the bare
# word would be a false positive. This is the single source of truth for
# what assert_blind checks; keep it in sync with the loop below.
_FORBIDDEN_BLIND_TERMS = ["tier 1", "tier 2", "tier 3", "tier 4", "baseline", "agent loop", "orchestrator"]


def assert_blind(candidate: str, prompt: str) -> None:
    """Defensive assertion: the candidate text passed to a judge prompt
    builder must carry no tier marker. This only checks the SHAPE of the
    call (no tier/baseline keyword literally concatenated in), it does not
    inspect the LLM prompt text itself (see build_prompt_3a /
    build_prompt_abstention for the actual blind prompt bodies)."""
    lowered = candidate.lower()
    for term in _FORBIDDEN_BLIND_TERMS:
        assert term not in lowered, f"candidate answer leaks tier marker {term!r}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _rate_pair(n_success: int, n_trials: int) -> tuple[int, int]:
    return n_success, n_trials


def score_all(
    root: Path,
    tickets_by_id: dict,
    tiers: list[int] | None,
    no_cache: bool,
) -> dict:
    records = discover_results(root, tiers)
    tiers_found = sorted({r["tier"] for r in records})
    repeats_found = {t: sorted({r["repeat"] for r in records if r["tier"] == t}) for t in tiers_found}

    cache_path = root / "judge_cache.json"
    cache = {} if no_cache else load_cache(cache_path)
    stats = JudgeStats()

    per_answer = []

    # primary[tier][repeat][category-or-"overall"] = (n_success, n_trials)
    primary_counts: dict[int, dict[int, dict[str, list[int]]]] = {}
    abstention_counts: dict[int, dict[int, list[int]]] = {}
    n_crashed: dict[int, int] = {}

    for rec in records:
        tier, repeat, ticket_id, result = rec["tier"], rec["repeat"], rec["ticket_id"], rec["result"]
        ticket = tickets_by_id.get(ticket_id)
        if ticket is None:
            continue
        expected_behavior = ticket.get("expected_behavior")
        category = ticket.get("category")
        ticket_text = ticket.get("ticket_text")
        gold_root_cause = ticket.get("gold_root_cause")

        answer = candidate_answer(result)
        primary_counts.setdefault(tier, {}).setdefault(repeat, {})
        abstention_counts.setdefault(tier, {}).setdefault(repeat, [0, 0])
        n_crashed.setdefault(tier, 0)

        if expected_behavior != "escalate":
            bucket = primary_counts[tier][repeat].setdefault(category, [0, 0])
            overall = primary_counts[tier][repeat].setdefault("overall", [0, 0])
            if answer is None:
                n_crashed[tier] += 1
                bucket[1] += 1
                overall[1] += 1
                per_answer.append(
                    {
                        "ticket_id": ticket_id,
                        "tier": tier,
                        "repeat": repeat,
                        "kind": "semantic_match",
                        "verdict": None,
                        "reasoning": "crashed / no answer",
                    }
                )
                continue
            assert_blind(answer, "")
            verdict, reasoning = judge_semantic(
                ticket_id, ticket_text, category, answer, gold_root_cause, cache, stats, no_cache
            )
            bucket[1] += 1
            overall[1] += 1
            if verdict is True:
                bucket[0] += 1
                overall[0] += 1
            per_answer.append(
                {
                    "ticket_id": ticket_id,
                    "tier": tier,
                    "repeat": repeat,
                    "kind": "semantic_match",
                    "verdict": verdict,
                    "reasoning": reasoning,
                }
            )
        else:
            abs_bucket = abstention_counts[tier][repeat]
            if answer is None:
                n_crashed[tier] += 1
                abs_bucket[1] += 1
                per_answer.append(
                    {
                        "ticket_id": ticket_id,
                        "tier": tier,
                        "repeat": repeat,
                        "kind": "abstention",
                        "verdict": None,
                        "reasoning": "crashed / no answer",
                    }
                )
                continue
            assert_blind(answer, "")
            verdict, reasoning = judge_abstention(ticket_id, ticket_text, answer, cache, stats, no_cache)
            abs_bucket[1] += 1
            if verdict is True:
                abs_bucket[0] += 1
            per_answer.append(
                {
                    "ticket_id": ticket_id,
                    "tier": tier,
                    "repeat": repeat,
                    "kind": "abstention",
                    "verdict": verdict,
                    "reasoning": reasoning,
                }
            )

    if not no_cache:
        save_cache(cache_path, cache)

    # Build cells: tier -> category-or-overall -> aggregate over repeats
    primary_cells: dict[int, dict[str, dict]] = {}
    for tier, by_repeat in primary_counts.items():
        repeats_sorted = sorted(by_repeat.keys())
        keys = set()
        for r in repeats_sorted:
            keys.update(by_repeat[r].keys())
        primary_cells[tier] = {}
        for key in keys:
            pairs = [tuple(by_repeat[r].get(key, [0, 0])) for r in repeats_sorted]
            primary_cells[tier][key] = aggregate_cell(pairs)

    abstention_cells: dict[int, dict] = {}
    for tier, by_repeat in abstention_counts.items():
        repeats_sorted = sorted(by_repeat.keys())
        pairs = [tuple(by_repeat[r]) for r in repeats_sorted]
        abstention_cells[tier] = aggregate_cell(pairs)

    return {
        "records": records,
        "tiers_found": tiers_found,
        "repeats_found": repeats_found,
        "primary_cells": primary_cells,
        "abstention_cells": abstention_cells,
        "n_crashed": n_crashed,
        "per_answer": per_answer,
        "n_judge_calls_made": stats.n_judge_calls_made,
        "n_cache_hits": stats.n_cache_hits,
    }


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def build_comparisons(primary_cells: dict, abstention_cells: dict) -> dict:
    comparisons = {"primary": {}, "abstention": None}
    if 4 not in primary_cells:
        return comparisons
    cell4 = primary_cells[4]
    for tier, cells in primary_cells.items():
        if tier == 4:
            continue
        comparisons["primary"][tier] = {}
        keys = set(cell4.keys()) | set(cells.keys())
        for key in keys:
            c4 = cell4.get(key, aggregate_cell([]))
            co = cells.get(key, aggregate_cell([]))
            comparisons["primary"][tier][key] = compare_to_tier4(c4, co)

    if 4 in abstention_cells:
        abs4 = abstention_cells[4]
        comparisons["abstention"] = {}
        for tier, cell in abstention_cells.items():
            if tier == 4:
                continue
            comparisons["abstention"][tier] = compare_to_tier4(abs4, cell)

    return comparisons


def all_inconclusive(comparisons: dict) -> bool:
    verdicts = []
    for by_tier in comparisons.get("primary", {}).values():
        for cell in by_tier.values():
            verdicts.append(cell["verdict"])
    if comparisons.get("abstention"):
        for cell in comparisons["abstention"].values():
            verdicts.append(cell["verdict"])
    if not verdicts:
        return False
    return all(v == "inconclusive" for v in verdicts)


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def _stat3(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "p50": None, "p95": None, "n": 0}
    ordered = sorted(clean)

    def _pct(p):
        idx = round(p * (len(ordered) - 1))
        return ordered[idx]

    return {"mean": statistics.mean(clean), "p50": _pct(0.5), "p95": _pct(0.95), "n": len(clean)}


def build_efficiency(records: list[dict], rate_card_label: str, rate_card) -> dict:
    by_tier: dict[int, list[dict]] = {}
    for rec in records:
        by_tier.setdefault(rec["tier"], []).append(rec["result"])

    out = {"rate_card": {"provider": rate_card_label, "used": rate_card is not None}, "tiers": {}}
    for tier, results in sorted(by_tier.items()):
        wall = [r.get("run", {}).get("wall_clock_seconds") for r in results]
        calls = [r.get("usage", {}).get("llm_call_count") for r in results]
        costs = []
        for r in results:
            usage = r.get("usage") or {}
            if rate_card is not None:
                costs.append(metrics.estimated_cost_usd(usage, rate_card))
            else:
                costs.append(None)
        total_cost = sum(c for c in costs if c is not None) if rate_card is not None else None
        out["tiers"][tier] = {
            "wall_clock_seconds": _stat3(wall),
            "estimated_cost_usd": _stat3(costs) if rate_card is not None else {"mean": None, "p50": None, "p95": None, "n": 0},
            "llm_call_count": _stat3(calls),
            "total_cost_usd": total_cost,
        }
    return out


# ---------------------------------------------------------------------------
# Tier-4-only section
# ---------------------------------------------------------------------------


def build_tier4_only(records: list[dict], tickets_by_id: dict) -> dict:
    tier4_results = [(rec["ticket_id"], rec["result"]) for rec in records if rec["tier"] == 4]

    def _values_for(name: str):
        out = []
        for ticket_id, result in tier4_results:
            ticket = tickets_by_id.get(ticket_id)
            if ticket is None:
                continue
            state = result.get("state")
            if name == "task_success_status_only":
                out.append(metrics.task_success_status_only(state, ticket))
            elif name == "task_success_strict_lexical":
                out.append(metrics.task_success_strict_lexical(state, ticket))
            elif name == "correct_escalation":
                if ticket.get("expected_behavior") == "escalate":
                    out.append(metrics.correct_escalation(state, ticket))
            elif name == "tool_selection_accuracy":
                v = metrics.tool_selection_accuracy(state, ticket)
                if v is not None:
                    out.append(v)
            elif name == "unnecessary_tool_calls":
                out.append(metrics.unnecessary_tool_calls(state, ticket))
            elif name == "parameter_validity":
                pv = metrics.parameter_validity(state)
                out.append(pv["observed_failures"])
            elif name == "state_consistency":
                out.append(metrics.state_consistency(state))
            elif name == "write_gate_appended_correctly":
                v = metrics.write_gate_appended_correctly(state)
                if v is not None:
                    out.append(v)
            elif name == "citation_presence":
                v = metrics.citation_presence(state)
                if v is not None:
                    out.append(v)
            elif name == "retrieval_recall_at_3_observed":
                v = metrics.retrieval_recall_at_3_observed(state, ticket)
                if v is not None:
                    out.append(v)
        return out

    tier4_only = {}
    for name in TIER4_ONLY_METRICS:
        values = _values_for(name)
        bool_values = [v for v in values if isinstance(v, bool)]
        num_values = [v for v in values if not isinstance(v, bool) and isinstance(v, (int, float))]
        if bool_values:
            entry = {"n_true": sum(bool_values), "n": len(bool_values), "rate": (sum(bool_values) / len(bool_values)) if bool_values else None}
        elif num_values:
            entry = {"sum": sum(num_values), "n": len(num_values), "mean": statistics.mean(num_values) if num_values else None}
        else:
            entry = {"n": 0, "rate": None}
        entry["tiers_1_3"] = "n/a (no TaskState)"
        tier4_only[name] = entry
    return tier4_only


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run(root: Path, out_path: Path, tiers: list[int] | None, no_cache: bool) -> dict:
    tickets_by_id = load_tickets()
    scored = score_all(root, tickets_by_id, tiers, no_cache)

    rate_card_label, rate_card = report.select_rate_card(os.environ.get("LLM_BASE_URL"))

    comparisons = build_comparisons(scored["primary_cells"], scored["abstention_cells"])
    inconclusive_everywhere = all_inconclusive(comparisons)

    efficiency = build_efficiency(scored["records"], rate_card_label, rate_card)
    tier4_only = build_tier4_only(scored["records"], tickets_by_id)

    summary_lines = []
    if inconclusive_everywhere:
        summary_lines.append(
            "Every category/overall/abstention comparison came back inconclusive at this "
            "sample size -- the comparison is inconclusive at this sample size. Medians below "
            "describe this run's data, they are not findings about which tier is better."
        )
    else:
        for tier, by_key in comparisons.get("primary", {}).items():
            for key, cell in by_key.items():
                if cell["verdict"] != "inconclusive":
                    summary_lines.append(f"tier{tier} vs tier4 [{key}]: {cell['verdict']} -- {cell['detail']}")
        if comparisons.get("abstention"):
            for tier, cell in comparisons["abstention"].items():
                if cell["verdict"] != "inconclusive":
                    summary_lines.append(f"tier{tier} vs tier4 [abstention]: {cell['verdict']} -- {cell['detail']}")
    summary = " ".join(summary_lines) if summary_lines else "No comparable cells were found."

    output = {
        "schema_version": 1,
        "config": {
            "root": str(root),
            "tiers": scored["tiers_found"],
            "repeats_found": scored["repeats_found"],
            "prompt_variant": llm_judge.PROMPT_VARIANT_3A,
            "abstention_prompt_variant": llm_judge.PROMPT_VARIANT_ABSTENTION,
            "rubric_version": llm_judge.RUBRIC_VERSION,
            "model": llm_judge.MODEL,
            "judge_temperature": 0.0,
            "n_judge_calls_made": scored["n_judge_calls_made"],
            "n_cache_hits": scored["n_cache_hits"],
        },
        "rate_card": {"provider": rate_card_label, "used": rate_card is not None},
        "primary": {str(tier): cells for tier, cells in scored["primary_cells"].items()},
        "abstention": {str(tier): cell for tier, cell in scored["abstention_cells"].items()},
        "n_crashed": {str(tier): n for tier, n in scored["n_crashed"].items()},
        "comparisons": {
            "primary": {str(tier): {str(k): v for k, v in cells.items()} for tier, cells in comparisons["primary"].items()},
            "abstention": (
                {str(tier): v for tier, v in comparisons["abstention"].items()}
                if comparisons.get("abstention")
                else None
            ),
        },
        "efficiency": efficiency,
        "tier4_only": tier4_only,
        "per_answer": scored["per_answer"],
        "summary": summary,
        "inconclusive_everywhere": inconclusive_everywhere,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return output


def print_summary(output: dict) -> None:
    print(output["summary"])
    print(
        f"n_judge_calls_made={output['config']['n_judge_calls_made']} "
        f"n_cache_hits={output['config']['n_cache_hits']}"
    )
    for tier, cells in output["primary"].items():
        overall = cells.get("overall", {})
        print(
            f"tier{tier}: overall median={overall.get('median')} "
            f"min={overall.get('min')} max={overall.get('max')}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--tiers", type=str, default=None, help="Comma-separated tier list, e.g. 1,2,3,4")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    tiers = [int(t.strip()) for t in args.tiers.split(",")] if args.tiers else None

    if not args.live:
        records = discover_results(root, tiers) if root.exists() else []
        n_answers = sum(1 for r in records if candidate_answer(r["result"]) is not None)
        print(
            "Refusing to run without --live: this calls the LLM judge once per "
            "(ticket, tier, repeat) answer (twice for the 10 ambiguous tickets -- "
            "semantic_match plus a separate abstention judgment). The current input "
            f"at {root} implies roughly {n_answers} judge call(s) (before caching). "
            "Pass --live to proceed.",
            file=sys.stderr,
        )
        return 2

    output = run(root, Path(args.out), tiers, args.no_cache)
    print_summary(output)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
