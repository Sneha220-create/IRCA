"""Leakage audit: quantify how much of the gold root cause is given away by
the ticket text itself.

Session 11's 4-tier comparison found that a plain LLM with no tools and no
RAG scores 96.2% because the ticket text states the diagnosis outright. This
module turns that finding into a reproducible, committed artifact so the
benchmark rewrite that follows it can be verified against a known baseline.

This is READ-ONLY analysis. It makes NO LLM calls and NO network requests --
it only reads data/tickets.json, tools/fake_data.py's per-ticket signatures,
and (optionally) an existing eval/results/tier_compare.json. There is no
--live gate because there is nothing live to gate.

Scope
-----
Only tickets where category != "adversarial" and expected_behavior !=
"escalate" are audited (53 of the current ticket set). Ambiguous/escalate
tickets are excluded because "the ticket text doesn't determine the answer"
is their intended design -- failing to resolve from text alone is correct
behavior for them, not evidence of resistance to leakage.

Classification is driven by the EMPIRICAL measure (tier 1 plain-LLM success
rate from eval/results/tier_compare.json), not by lexical token overlap.
Lexical overlap (gold_token_hits / metric_token_hits / giveaway_spans) is
supporting evidence for a human doing the ticket rewrite -- it points at the
literal phrases to remove -- but it is not what decides the classification.
A ticket can leak semantically (paraphrase, closely related wording) without
sharing a single token with the gold root-cause slug; five such tickets
exist in the current set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from eval.run_benchmark import tickets_content_sha256
from tools import fake_data

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"
DEFAULT_TIER_COMPARE_PATH = REPO_ROOT / "eval" / "results" / "tier_compare.json"
DEFAULT_OUT_PATH = REPO_ROOT / "eval" / "results" / "leakage_audit.json"

CONTEXT_CHARS = 40
MIN_TOKEN_LEN = 3  # tokens of length <= this are dropped, not matched
PREFIX_LEN = 6  # tolerant inflection match: compare first N chars of token vs word


def _tokens_from_slug(slug: str) -> list[str]:
    """Split a gold_root_cause / metric_name slug on underscores, keeping
    only tokens longer than MIN_TOKEN_LEN characters."""
    return [t for t in slug.split("_") if len(t) > MIN_TOKEN_LEN]


_WORD_RE = re.compile(r"[A-Za-z]+")


def _find_token_matches(token: str, text: str) -> list[tuple[int, int, str]]:
    """Find matches of `token` in `text`, tolerant of simple plural /
    inflection differences (e.g. "connection" matching "connections",
    "rotating" matching "rotation"). Matching rule: a word in the text
    matches the token if, case-insensitively, either is a prefix of the
    other using the first PREFIX_LEN characters of the token -- i.e. the
    first min(PREFIX_LEN, len(token)) characters of the token must equal
    the first same-many characters of the candidate word. This is the rule
    the exploratory pass used and is deliberately loose (a prefix match) to
    catch plural/gerund/tense variants without a full stemmer.

    Returns a list of (start, end, matched_substring) spans in the
    original-cased text.
    """
    token_lower = token.lower()
    prefix = token_lower[:PREFIX_LEN]
    matches = []
    for m in _WORD_RE.finditer(text):
        word = m.group(0)
        word_lower = word.lower()
        if len(word_lower) <= MIN_TOKEN_LEN:
            continue
        word_prefix = word_lower[:PREFIX_LEN]
        if word_prefix == prefix or word_lower.startswith(token_lower) or token_lower.startswith(word_lower):
            matches.append((m.start(), m.end(), word))
    return matches


def _span_with_context(text: str, start: int, end: int) -> str:
    ctx_start = max(0, start - CONTEXT_CHARS)
    ctx_end = min(len(text), end + CONTEXT_CHARS)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return f"{prefix}{text[ctx_start:ctx_end]}{suffix}"


def _token_hits_and_spans(tokens: list[str], text: str) -> tuple[list[str], list[str]]:
    hits: list[str] = []
    spans: list[str] = []
    seen_tokens: set[str] = set()
    for token in tokens:
        matches = _find_token_matches(token, text)
        if not matches:
            continue
        if token not in seen_tokens:
            hits.append(token)
            seen_tokens.add(token)
        for start, end, word in matches:
            spans.append(_span_with_context(text, start, end))
    return hits, spans


def _load_tier1_stats(tier_compare_path: Path) -> dict[str, dict] | None:
    """Return {ticket_id: {"successes": int, "trials": int}} built from
    tier 1, non-abstention entries in tier_compare.json's per_answer list.
    Returns None if the file is missing (empirical fields stay None)."""
    if not tier_compare_path.exists():
        return None
    data = json.loads(tier_compare_path.read_text(encoding="utf-8"))
    stats: dict[str, dict] = {}
    for entry in data.get("per_answer", []):
        if entry.get("tier") != 1:
            continue
        if entry.get("kind") == "abstention":
            continue
        ticket_id = entry["ticket_id"]
        bucket = stats.setdefault(ticket_id, {"successes": 0, "trials": 0})
        bucket["trials"] += 1
        if entry.get("verdict"):
            bucket["successes"] += 1
    return stats


def _classify(tier1_rate: float | None) -> str:
    if tier1_rate is None:
        return "unknown"
    if tier1_rate == 1.0:
        return "leaked_strong"
    if tier1_rate == 0.0:
        return "resistant"
    return "leaked_partial"


def audit_tickets(tickets_path: Path, tier_compare_path: Path) -> dict:
    tickets = json.loads(Path(tickets_path).read_text(encoding="utf-8"))
    scoped = [
        t for t in tickets
        if t.get("category") != "adversarial" and t.get("expected_behavior") != "escalate"
    ]

    tier1_stats = _load_tier1_stats(Path(tier_compare_path))
    tier_compare_available = tier1_stats is not None

    rows = []
    for ticket in scoped:
        ticket_id = ticket["id"]
        text = ticket["ticket_text"]
        gold_slug = ticket["gold_root_cause"]

        signature = fake_data.signature_for_ticket(ticket_id) or {}
        metric_name = signature.get("metric_name", "")

        gold_tokens = _tokens_from_slug(gold_slug)
        metric_tokens = _tokens_from_slug(metric_name) if metric_name else []

        gold_hits, gold_spans = _token_hits_and_spans(gold_tokens, text)
        metric_hits, metric_spans = _token_hits_and_spans(metric_tokens, text)

        giveaway_spans = gold_spans + metric_spans

        if tier_compare_available:
            bucket = tier1_stats.get(ticket_id)
            if bucket is None or bucket["trials"] == 0:
                successes = trials = None
                rate = None
            else:
                successes = bucket["successes"]
                trials = bucket["trials"]
                rate = successes / trials
        else:
            successes = trials = rate = None

        classification = _classify(rate)

        rows.append({
            "ticket_id": ticket_id,
            "category": ticket["category"],
            "gold_root_cause": gold_slug,
            "tier1_successes": successes,
            "tier1_trials": trials,
            "tier1_rate": rate,
            "gold_token_hits": gold_hits,
            "metric_token_hits": metric_hits,
            "giveaway_spans": giveaway_spans,
            "classification": classification,
        })

    aggregate = _build_aggregate(rows, tier_compare_available)

    tickets_path = Path(tickets_path)
    return {
        "config": {
            "tickets_path": str(tickets_path),
            "tickets_sha256": tickets_content_sha256(tickets_path),
        },
        "tier_compare_available": tier_compare_available,
        "audited_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
    }


def _build_aggregate(rows: list[dict], tier_compare_available: bool) -> dict:
    classifications = ["leaked_strong", "leaked_partial", "resistant", "unknown"]
    overall_counts = {c: 0 for c in classifications}
    per_category_counts: dict[str, dict[str, int]] = {}
    crosstab: dict[str, dict[str, int]] = {
        c: {"has_lexical_overlap": 0, "no_lexical_overlap": 0} for c in classifications
    }
    n_with_lexical_overlap = 0

    for row in rows:
        c = row["classification"]
        overall_counts[c] += 1

        cat = row["category"]
        per_category_counts.setdefault(cat, {k: 0 for k in classifications})
        per_category_counts[cat][c] += 1

        has_overlap = bool(row["gold_token_hits"] or row["metric_token_hits"])
        if has_overlap:
            n_with_lexical_overlap += 1
            crosstab[c]["has_lexical_overlap"] += 1
        else:
            crosstab[c]["no_lexical_overlap"] += 1

    audited = len(rows)
    leak_rate = (overall_counts["leaked_strong"] / audited) if audited else None

    return {
        "overall_counts": overall_counts,
        "per_category_counts": per_category_counts,
        "n_with_lexical_overlap": n_with_lexical_overlap,
        "classification_x_lexical_overlap": crosstab,
        "leak_rate": leak_rate,
        "note": None if tier_compare_available else (
            "eval/results/tier_compare.json was not found; empirical fields "
            "(tier1_*, classification) are None/\"unknown\" for all rows."
        ),
    }


def _persist_result(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / f".{out_path.name}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _write_markdown(result: dict, md_path: Path) -> None:
    lines = [
        "| ticket_id | category | gold_root_cause | tier1_rate | classification | gold_token_hits | metric_token_hits |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            "| {ticket_id} | {category} | {gold_root_cause} | {tier1_rate} | {classification} | {gold} | {metric} |".format(
                ticket_id=row["ticket_id"],
                category=row["category"],
                gold_root_cause=row["gold_root_cause"],
                tier1_rate=row["tier1_rate"],
                classification=row["classification"],
                gold=", ".join(row["gold_token_hits"]) or "-",
                metric=", ".join(row["metric_token_hits"]) or "-",
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(result: dict) -> None:
    agg = result["aggregate"]
    print(f"Audited {result['audited_count']} tickets.")
    if not result["tier_compare_available"]:
        print("NOTE:", agg["note"])
    print("Overall counts:", agg["overall_counts"])
    print("Per-category counts:", agg["per_category_counts"])
    print("n_with_lexical_overlap:", agg["n_with_lexical_overlap"])
    print("classification x lexical overlap:", agg["classification_x_lexical_overlap"])
    print(f"leak_rate: {agg['leak_rate']}")
    print()
    print("Non-leaked_strong tickets (rewrite candidates):")
    for row in result["rows"]:
        if row["classification"] != "leaked_strong":
            print(
                f"  {row['ticket_id']} [{row['category']}] "
                f"classification={row['classification']} tier1_rate={row['tier1_rate']} "
                f"gold_root_cause={row['gold_root_cause']}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit ticket text for gold root-cause leakage.")
    # Resolved lazily (default=None -> fake_data.resolve_tickets_path() at
    # call time) so IRCA_TICKETS_PATH set after import/parser-construction
    # is still honoured.
    parser.add_argument("--tickets", type=Path, default=None)
    parser.add_argument("--tier-compare", type=Path, default=DEFAULT_TIER_COMPARE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    tickets_path = args.tickets if args.tickets is not None else fake_data.resolve_tickets_path()
    result = audit_tickets(tickets_path, args.tier_compare)
    _persist_result(result, args.out)
    if args.markdown:
        _write_markdown(result, args.markdown)

    _print_summary(result)


if __name__ == "__main__":
    main()
