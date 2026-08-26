"""Validate a symptom-only rewrite of data/tickets.json against docs/decisions.md
J5 and J6 before any paid sweep is run against it.

This is READ-ONLY analysis over two JSON files plus
`tools.fake_data.signature_for_ticket`. No LLM calls, no network, no cost.

Checks
------
Structural (fatal -- stop on first failure among these):
  S1 -- same ticket ids, same order, same count as baseline.
  S2 -- only `ticket_text` differs; every other field is byte-identical.
  S3 -- out-of-scope tickets (adversarial / expected_behavior == "escalate")
        are byte-identical, including ticket_text.

Per in-scope ticket (category != "adversarial" and
expected_behavior != "escalate"):
  L1 -- no gold-slug vocabulary left in ticket_text.
  L2 -- no metric-name vocabulary left in ticket_text.
  L3 -- no telemetry-source references (log, dashboard, metric, alert, ...)
        left in ticket_text. ABSOLUTE, not differential (see below). A
        ticket is supposed to report the COMPLAINT (what someone noticed),
        not the INVESTIGATION (what the telemetry says once someone looked).
        A rewrite that keeps phrasing like "the dashboard shows..." or
        "per the metrics" still leaks the diagnosis path even when it swaps
        out the gold slug and metric-name tokens L1/L2 catch -- the
        investigation reading itself belongs to what `tools/fake_data.py`
        returns when the agent calls a tool, not to the ticket the agent
        starts from. A ticket that quotes it re-introduces the leak in a
        form L1/L2 cannot detect.
  D1 -- the gold cause is still derivable from the ticket's fake_data
        signature (log_patterns + metric_name). DIFFERENTIAL: computed on
        both baseline and candidate; FAILs only where the candidate is
        newly not-derivable and the baseline WAS derivable (a regression
        the rewrite introduced). Where both are not-derivable it is
        reported as pre_existing (INFO), never FAIL.
  W1 -- length sanity (WARN only).
  W2 -- rewritten text still describes an observable (WARN only).
        DIFFERENTIAL like D1: warns only where the candidate lost an
        observable the baseline had; a ticket that never had one is
        pre_existing, not a warning.

Why D1 and W2 are differential, not absolute
---------------------------------------------
An early absolute-threshold run of D1 FAILed on the unmodified BASELINE for
9 tickets -- the whole `deploy_healthcheck_misconfiguration` cluster (T004,
T008, T013, T019, T025, T029, T035, T044, T051). Their signature describes
"readiness probe failed" / HTTP 503 probe failures / a
`readiness_probe_success_rate` metric, which shares no 6-char-prefix token
with the gold slug's "deploy"/"healthcheck"/"misconfiguration" -- a false
alarm from lexical matching, not real unsolvability (an engineer reading
"readiness probe failing" does diagnose a misconfigured readiness probe;
see eval/llm_judge.py's RUBRIC_TEXT clause (a)). A check that fails
identically on baseline-as-candidate cannot tell us anything about a
rewrite, so both checks now report only what the rewrite changed.

Because a rewrite only touches `ticket_text` and D1 reads solely the
fake_data SIGNATURE (log_patterns + metric_name), D1's per-ticket result
cannot actually change between baseline and candidate today -- it will
only ever report pre_existing findings here. It is kept anyway: it is a
real, useful finding about `tools/fake_data.py`'s signatures, and it
guards against a future change that does touch signatures. Do not
"tighten" D1 or W2 back into absolute checks -- that reintroduces the
readiness-probe false positive.

Why L3 is absolute, not differential
-------------------------------------
Unlike D1/W2, L3 does not compare candidate against baseline. The baseline
tickets are expected to fail L3 heavily -- that IS the leak this whole
harness exists to catch, exactly like L1/L2. A differential version would
only report telemetry references the rewrite *added*, which misses the
entire point: the goal is a ticket with zero telemetry-source references,
not merely no worse than the baseline.

The lexical matcher (token splitting + inflection-tolerant span finding) is
imported from eval.leakage_audit rather than reimplemented.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval.leakage_audit import (
    _find_token_matches,
    _span_with_context,
    _tokens_from_slug,
)
from tools import fake_data

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = REPO_ROOT / "data" / "tickets.json"

W1_MIN_LEN = 40
W1_MIN_RATIO = 0.4

# Telemetry-source vocabulary that must never appear in ticket_text: a ticket
# reports the COMPLAINT (what someone noticed), never the INVESTIGATION
# (what the telemetry says once someone looked). Terms below are the words
# tickets use to cite dashboards/logs/metrics/alerts directly -- readings
# that belong in what tools/fake_data.py returns when the agent calls a
# tool, not in the ticket it starts from. Edit this list here, in one place,
# if a new telemetry-citing term shows up in a future rewrite attempt.
TELEMETRY_TERMS = [
    "log", "logs", "logging",
    "dashboard", "dashboards",
    "metric", "metrics",
    "graph", "graphs",
    "chart", "charts",
    "telemetry",
    "alert", "alerts",
    "monitor", "monitoring", "monitors",
    "trace", "traces",
    "probe", "probes",
    "gauge", "counter", "histogram",
]

_TELEMETRY_TERM_RE = {
    term: re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in TELEMETRY_TERMS
}

_DIGIT_RE = re.compile(r"\d")
_PERCENT_RE = re.compile(r"%")
_HTTP_STATUS_RE = re.compile(r"\b[1-5]\d{2}\b")
_TIME_REF_RE = re.compile(
    r"\b\d{1,2}:\d{2}\b|\b(am|pm|AM|PM)\b|\bminutes?\b|\bhours?\b|\bmin\b|\bsec\b|\bseconds?\b|\bago\b"
)


def _out_of_scope(ticket: dict) -> bool:
    return ticket.get("category") == "adversarial" or ticket.get("expected_behavior") == "escalate"


def _load(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_s1(baseline: list[dict], candidate: list[dict]) -> dict:
    baseline_ids = [t["id"] for t in baseline]
    candidate_ids = [t["id"] for t in candidate]
    detail: list[str] = []
    passed = True

    if len(baseline_ids) != len(candidate_ids):
        passed = False
        detail.append(f"count mismatch: baseline={len(baseline_ids)} candidate={len(candidate_ids)}")

    baseline_set = set(baseline_ids)
    candidate_set = set(candidate_ids)
    added = candidate_set - baseline_set
    removed = baseline_set - candidate_set
    if added:
        passed = False
        detail.append(f"ids added not in baseline: {sorted(added)}")
    if removed:
        passed = False
        detail.append(f"ids removed from baseline: {sorted(removed)}")

    if not added and not removed and baseline_ids != candidate_ids:
        passed = False
        detail.append(
            f"order mismatch: baseline order={baseline_ids} candidate order={candidate_ids}"
        )

    return {"check": "S1", "name": "same ids/order/count", "passed": passed, "detail": detail}


def _check_s2(baseline: list[dict], candidate: list[dict]) -> dict:
    detail: list[str] = []
    passed = True
    baseline_by_id = {t["id"]: t for t in baseline}
    candidate_by_id = {t["id"]: t for t in candidate}

    for ticket_id, cand_ticket in candidate_by_id.items():
        base_ticket = baseline_by_id.get(ticket_id)
        if base_ticket is None:
            continue  # reported by S1
        all_fields = set(base_ticket.keys()) | set(cand_ticket.keys())
        for field in sorted(all_fields):
            if field == "ticket_text":
                continue
            base_val = base_ticket.get(field)
            cand_val = cand_ticket.get(field)
            if base_val != cand_val:
                passed = False
                detail.append(
                    f"{ticket_id}: field '{field}' changed: baseline={base_val!r} candidate={cand_val!r}"
                )

    return {"check": "S2", "name": "only ticket_text differs", "passed": passed, "detail": detail}


def _check_s3(baseline: list[dict], candidate: list[dict]) -> dict:
    detail: list[str] = []
    passed = True
    baseline_by_id = {t["id"]: t for t in baseline}
    candidate_by_id = {t["id"]: t for t in candidate}

    for ticket_id, base_ticket in baseline_by_id.items():
        if not _out_of_scope(base_ticket):
            continue
        cand_ticket = candidate_by_id.get(ticket_id)
        if cand_ticket is None:
            continue  # reported by S1
        if base_ticket != cand_ticket:
            passed = False
            diff_fields = sorted(
                f for f in set(base_ticket.keys()) | set(cand_ticket.keys())
                if base_ticket.get(f) != cand_ticket.get(f)
            )
            detail.append(
                f"{ticket_id}: out-of-scope ticket changed, fields={diff_fields}"
            )

    return {"check": "S3", "name": "out-of-scope tickets byte-identical", "passed": passed, "detail": detail}


def _signature_text(ticket_id: str) -> tuple[list[str], str]:
    """Return (log_patterns, metric_name) for a ticket, tolerant of a
    missing/None signature."""
    signature = fake_data.signature_for_ticket(ticket_id) or {}
    log_patterns = signature.get("log_patterns") or []
    metric_name = signature.get("metric_name") or ""
    return log_patterns, metric_name


def _check_l1_l2(in_scope: list[dict]) -> tuple[dict, dict]:
    l1_detail: list[str] = []
    l2_detail: list[str] = []
    l1_passed = True
    l2_passed = True

    for ticket in in_scope:
        ticket_id = ticket["id"]
        text = ticket["ticket_text"]
        gold_slug = ticket["gold_root_cause"]
        log_patterns, metric_name = _signature_text(ticket_id)

        gold_tokens = _tokens_from_slug(gold_slug) if gold_slug else []
        for token in gold_tokens:
            matches = _find_token_matches(token, text)
            if matches:
                l1_passed = False
                for start, end, word in matches:
                    span = _span_with_context(text, start, end)
                    l1_detail.append(
                        f"{ticket_id}: gold token '{token}' matched word '{word}' -> {span}"
                    )

        metric_tokens = _tokens_from_slug(metric_name) if metric_name else []
        for token in metric_tokens:
            matches = _find_token_matches(token, text)
            if matches:
                l2_passed = False
                for start, end, word in matches:
                    span = _span_with_context(text, start, end)
                    l2_detail.append(
                        f"{ticket_id}: metric token '{token}' matched word '{word}' -> {span}"
                    )

    return (
        {"check": "L1", "name": "no gold-slug vocabulary in ticket_text", "passed": l1_passed, "detail": l1_detail},
        {"check": "L2", "name": "no metric-name vocabulary in ticket_text", "passed": l2_passed, "detail": l2_detail},
    )


def _check_l3(in_scope: list[dict]) -> dict:
    """ABSOLUTE (not differential): FAIL on any whole-word telemetry-source
    reference in ticket_text. See module docstring for why this is absolute
    and why it exists (a ticket must report the complaint, not quote the
    investigation)."""
    detail: list[str] = []
    passed = True

    for ticket in in_scope:
        ticket_id = ticket["id"]
        text = ticket["ticket_text"]
        for term, pattern in _TELEMETRY_TERM_RE.items():
            for match in pattern.finditer(text):
                passed = False
                span = _span_with_context(text, match.start(), match.end())
                detail.append(
                    f"{ticket_id}: telemetry term '{term}' matched '{match.group(0)}' -> {span}"
                )

    return {
        "check": "L3",
        "name": "no telemetry-source references in ticket_text",
        "passed": passed,
        "detail": detail,
    }


def _ticket_derivable(ticket: dict) -> bool:
    """True if at least one gold-slug token appears in the ticket's
    fake_data signature evidence (log_patterns + metric_name)."""
    ticket_id = ticket["id"]
    gold_slug = ticket["gold_root_cause"]
    if not gold_slug:
        return True
    log_patterns, metric_name = _signature_text(ticket_id)
    evidence_text = " ".join(log_patterns) + " " + metric_name
    gold_tokens = _tokens_from_slug(gold_slug)
    for token in gold_tokens:
        if _find_token_matches(token, evidence_text):
            return True
    return not gold_tokens  # no tokens to check -> vacuously derivable


def _check_d1(in_scope_candidate: list[dict], baseline_by_id: dict[str, dict]) -> dict:
    """Differential: FAIL only where the candidate is not-derivable and the
    baseline (same ticket id) WAS derivable -- a regression the rewrite
    introduced. Where both are not-derivable, report pre_existing (INFO),
    never FAIL. See module docstring for why this is differential."""
    regression_detail: list[str] = []
    pre_existing_detail: list[str] = []

    for ticket in in_scope_candidate:
        if _ticket_derivable(ticket):
            continue
        ticket_id = ticket["id"]
        log_patterns, metric_name = _signature_text(ticket_id)
        base_ticket = baseline_by_id.get(ticket_id)
        baseline_derivable = _ticket_derivable(base_ticket) if base_ticket is not None else True
        line = (
            f"{ticket_id}: gold_root_cause={ticket['gold_root_cause']!r} not derivable; "
            f"signature log_patterns={log_patterns} metric_name={metric_name!r}"
        )
        if baseline_derivable:
            regression_detail.append(line)
        else:
            pre_existing_detail.append(line)

    return {
        "check": "D1",
        "name": "gold cause derivable from evidence (differential vs baseline)",
        "description": (
            "D1 reads only the fake_data signature (log_patterns + metric_name), which a "
            "ticket_text-only rewrite never touches, so D1's result cannot actually change "
            "between baseline and candidate -- it can only ever report pre_existing findings "
            "here. Kept to guard future changes that do touch signatures."
        ),
        "passed": not regression_detail,
        "regressions": len(regression_detail),
        "pre_existing": len(pre_existing_detail),
        "regression_detail": regression_detail,
        "pre_existing_detail": pre_existing_detail,
    }


def _check_w1(in_scope_candidate: list[dict], baseline_by_id: dict[str, dict]) -> dict:
    detail: list[str] = []
    warned = False
    for ticket in in_scope_candidate:
        ticket_id = ticket["id"]
        cand_text = ticket["ticket_text"]
        base_ticket = baseline_by_id.get(ticket_id)
        base_text = base_ticket["ticket_text"] if base_ticket else ""
        cand_len = len(cand_text)
        base_len = len(base_text)
        ratio = (cand_len / base_len) if base_len else None
        if cand_len < W1_MIN_LEN or (ratio is not None and ratio < W1_MIN_RATIO):
            warned = True
            detail.append(
                f"{ticket_id}: candidate length={cand_len} baseline length={base_len} "
                f"ratio={ratio if ratio is None else round(ratio, 3)}"
            )
    return {"check": "W1", "name": "length sanity", "warned": warned, "detail": detail}


def _describes_observable(text: str) -> bool:
    return bool(
        _DIGIT_RE.search(text)
        or _PERCENT_RE.search(text)
        or _HTTP_STATUS_RE.search(text)
        or _TIME_REF_RE.search(text)
    )


def _check_w2(in_scope_candidate: list[dict], baseline_by_id: dict[str, dict]) -> dict:
    """Differential: WARN only where the candidate lacks an observable and
    the baseline (same ticket id) HAD one -- the rewrite removed it. A
    ticket that never had an observable is pre_existing, not a warning."""
    regression_detail: list[str] = []
    pre_existing_detail: list[str] = []

    for ticket in in_scope_candidate:
        if _describes_observable(ticket["ticket_text"]):
            continue
        ticket_id = ticket["id"]
        base_ticket = baseline_by_id.get(ticket_id)
        base_had_observable = (
            _describes_observable(base_ticket["ticket_text"]) if base_ticket is not None else False
        )
        line = f"{ticket_id}: no digit/percent/http-status/time-reference found in ticket_text"
        if base_had_observable:
            regression_detail.append(line)
        else:
            pre_existing_detail.append(line)

    return {
        "check": "W2",
        "name": "text still describes an observable (differential vs baseline)",
        "warned": bool(regression_detail),
        "regressions": len(regression_detail),
        "pre_existing": len(pre_existing_detail),
        "regression_detail": regression_detail,
        "pre_existing_detail": pre_existing_detail,
    }


def validate(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)

    s1 = _check_s1(baseline, candidate)
    result: dict = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "structural_checks": [s1],
        "fatal": not s1["passed"],
    }
    if not s1["passed"]:
        result["lexical_checks"] = []
        result["derivability_checks"] = []
        result["warnings"] = []
        result["in_scope_count"] = None
        result["exit_ok"] = False
        return result

    s2 = _check_s2(baseline, candidate)
    s3 = _check_s3(baseline, candidate)
    result["structural_checks"] = [s1, s2, s3]

    if not s2["passed"] or not s3["passed"]:
        result["fatal"] = True
        result["lexical_checks"] = []
        result["derivability_checks"] = []
        result["warnings"] = []
        result["in_scope_count"] = None
        result["exit_ok"] = False
        return result

    result["fatal"] = False

    candidate_by_id = {t["id"]: t for t in candidate}
    baseline_by_id = {t["id"]: t for t in baseline}
    in_scope_candidate = [t for t in candidate if not _out_of_scope(t)]

    l1, l2 = _check_l1_l2(in_scope_candidate)
    l3 = _check_l3(in_scope_candidate)
    d1 = _check_d1(in_scope_candidate, baseline_by_id)
    w1 = _check_w1(in_scope_candidate, baseline_by_id)
    w2 = _check_w2(in_scope_candidate, baseline_by_id)

    result["lexical_checks"] = [l1, l2, l3]
    result["derivability_checks"] = [d1]
    result["warnings"] = [w1, w2]
    result["in_scope_count"] = len(in_scope_candidate)

    all_fail_checks = [l1, l2, l3, d1]
    result["exit_ok"] = all(c["passed"] for c in all_fail_checks)

    return result


def _print_check(check: dict) -> None:
    status = "PASS" if check["passed"] else "FAIL"
    print(f"[{status}] {check['check']}: {check['name']} (affected={len(check['detail'])})")
    for line in check["detail"]:
        print(f"    {line}")


def _print_warning(check: dict) -> None:
    status = "WARN" if check["warned"] else "PASS"
    print(f"[{status}] {check['check']}: {check['name']} (affected={len(check['detail'])})")
    for line in check["detail"]:
        print(f"    {line}")


def _print_differential(check: dict, warn_only: bool = False) -> None:
    """Print a differential (regression / pre_existing) check result."""
    if warn_only:
        status = "WARN" if check["regressions"] else "PASS"
    else:
        status = "FAIL" if check["regressions"] else "PASS"
    print(
        f"[{status}] {check['check']}: {check['name']} "
        f"(regressions={check['regressions']}, pre_existing={check['pre_existing']})"
    )
    if check.get("description"):
        print(f"    note: {check['description']}")
    for line in check["regression_detail"]:
        print(f"    REGRESSION: {line}")
    for line in check["pre_existing_detail"]:
        print(f"    pre_existing: {line}")


def _print_report(result: dict) -> None:
    print("=== Structural checks ===")
    for check in result["structural_checks"]:
        _print_check(check)

    if result["fatal"]:
        print()
        print("VERDICT: FAIL (structural check failed, stopping before per-ticket checks)")
        return

    print()
    print("=== Lexical checks (in-scope tickets) ===")
    for check in result["lexical_checks"]:
        _print_check(check)

    print()
    print("=== Derivability checks (differential vs baseline) ===")
    for check in result["derivability_checks"]:
        _print_differential(check)

    print()
    print("=== Warnings (non-fatal) ===")
    _print_warning(result["warnings"][0])
    _print_differential(result["warnings"][1], warn_only=True)

    regressions = sum(c.get("regressions", 0) for c in result["derivability_checks"])
    print()
    verdict = "PASS" if result["exit_ok"] else "FAIL"
    print(
        f"VERDICT: {verdict} -- {result['in_scope_count']} in-scope tickets validated, "
        f"{regressions} D1 regression(s) found"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a symptom-only ticket rewrite against docs/decisions.md J5/J6."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="print the JSON report instead of the readable one")
    args = parser.parse_args(argv)

    result = validate(args.baseline, args.candidate)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    return 0 if result["exit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
