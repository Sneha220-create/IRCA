import json
import os

import pytest

from eval import leakage_audit
from eval.run_benchmark import tickets_content_sha256


def _write_tickets(path, tickets):
    path.write_text(json.dumps(tickets), encoding="utf-8")


def _write_tier_compare(path, per_answer):
    path.write_text(json.dumps({"per_answer": per_answer}), encoding="utf-8")


def _base_ticket(**overrides):
    ticket = {
        "id": "T900",
        "ticket_text": "The database connections are pinned near max after a connection pool timeout.",
        "category": "easy",
        "gold_root_cause": "db_connection_pool_exhaustion",
        "gold_runbook_id": "RB-DB-001.md",
        "required_tools": ["query_logs"],
        "min_confidence_evidence_sources": 2,
        "expected_behavior": "resolve_with_approval",
        "notes": "",
    }
    ticket.update(overrides)
    return ticket


def test_classification_leaked_strong(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    per_answer = [
        {"ticket_id": "T900", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": True},
        {"ticket_id": "T900", "tier": 1, "repeat": 1, "kind": "semantic_match", "verdict": True},
    ]
    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, per_answer)

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    row = result["rows"][0]
    assert row["tier1_rate"] == 1.0
    assert row["classification"] == "leaked_strong"


def test_classification_leaked_partial(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    per_answer = [
        {"ticket_id": "T900", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": True},
        {"ticket_id": "T900", "tier": 1, "repeat": 1, "kind": "semantic_match", "verdict": False},
    ]
    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, per_answer)

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    row = result["rows"][0]
    assert row["tier1_rate"] == 0.5
    assert row["classification"] == "leaked_partial"


def test_classification_resistant(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    per_answer = [
        {"ticket_id": "T900", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": False},
        {"ticket_id": "T900", "tier": 1, "repeat": 1, "kind": "semantic_match", "verdict": False},
    ]
    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, per_answer)

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    row = result["rows"][0]
    assert row["tier1_rate"] == 0.0
    assert row["classification"] == "resistant"


def test_classification_unknown_when_no_tier1_entries(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    # Only tier 2 entries exist for this ticket -- no tier 1 data at all.
    per_answer = [
        {"ticket_id": "T900", "tier": 2, "repeat": 0, "kind": "semantic_match", "verdict": True},
    ]
    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, per_answer)

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    row = result["rows"][0]
    assert row["tier1_rate"] is None
    assert row["classification"] == "unknown"


def test_only_tier1_nonabstention_entries_drive_empirical_measure(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    per_answer = [
        # Tier 1, real signal: 1 success, 1 failure -> rate 0.5
        {"ticket_id": "T900", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": True},
        {"ticket_id": "T900", "tier": 1, "repeat": 1, "kind": "semantic_match", "verdict": False},
        # Tier 1 abstention -- must be ignored
        {"ticket_id": "T900", "tier": 1, "repeat": 2, "kind": "abstention", "verdict": False},
        # Tier 2/3/4 entries -- must be ignored regardless of verdict
        {"ticket_id": "T900", "tier": 2, "repeat": 0, "kind": "semantic_match", "verdict": True},
        {"ticket_id": "T900", "tier": 3, "repeat": 0, "kind": "semantic_match", "verdict": True},
        {"ticket_id": "T900", "tier": 4, "repeat": 0, "kind": "semantic_match", "verdict": True},
    ]
    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, per_answer)

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    row = result["rows"][0]
    assert row["tier1_trials"] == 2
    assert row["tier1_successes"] == 1
    assert row["tier1_rate"] == 0.5
    assert row["classification"] == "leaked_partial"


def test_scope_excludes_escalate_and_adversarial(tmp_path):
    tickets = [
        _base_ticket(id="T900", category="easy", expected_behavior="resolve_with_approval"),
        _base_ticket(id="T901", category="ambiguous", expected_behavior="escalate"),
        _base_ticket(id="T902", category="adversarial", expected_behavior="resolve_ignoring_injection"),
    ]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    tc_path = tmp_path / "does_not_exist.json"

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    ids = {row["ticket_id"] for row in result["rows"]}
    assert ids == {"T900"}
    assert result["audited_count"] == 1


def test_token_matching_inflection_and_min_length():
    text = "The connections are rotating oddly, and a db appeared briefly."
    hits, spans = leakage_audit._token_hits_and_spans(["connection", "rotation", "db"], text)
    assert "connection" in hits
    assert "rotation" in hits
    # "db" has length <= MIN_TOKEN_LEN (3) and must not be matched
    assert "db" not in hits


def test_token_matching_case_insensitive():
    text = "CONNECTIONS pinned near max."
    hits, spans = leakage_audit._token_hits_and_spans(["connection"], text)
    assert hits == ["connection"]


def test_giveaway_spans_present_with_overlap():
    text = "Error logs mention connection timeouts near the pool limit."
    hits, spans = leakage_audit._token_hits_and_spans(["connection"], text)
    assert len(spans) == 1
    assert "connection" in spans[0].lower()


def test_giveaway_spans_empty_without_overlap():
    text = "Nothing relevant appears in this ticket text at all."
    hits, spans = leakage_audit._token_hits_and_spans(["zzzznomatch"], text)
    assert hits == []
    assert spans == []


def test_no_lexical_overlap_but_tier1_perfect_is_still_leaked_strong(tmp_path):
    ticket = _base_ticket(
        id="T900",
        ticket_text="Users report the checkout flow is slow and occasionally fails outright.",
        gold_root_cause="db_connection_pool_exhaustion",
    )
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, [ticket])

    per_answer = [
        {"ticket_id": "T900", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": True},
        {"ticket_id": "T900", "tier": 1, "repeat": 1, "kind": "semantic_match", "verdict": True},
    ]
    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, per_answer)

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    row = result["rows"][0]
    assert row["gold_token_hits"] == []
    assert row["metric_token_hits"] == []
    assert row["classification"] == "leaked_strong"


def test_missing_tier_compare_file_runs_without_crash(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    missing_path = tmp_path / "no_such_file.json"

    result = leakage_audit.audit_tickets(tickets_path, missing_path)
    assert result["tier_compare_available"] is False
    row = result["rows"][0]
    assert row["tier1_rate"] is None
    assert row["tier1_successes"] is None
    assert row["tier1_trials"] is None
    assert row["classification"] == "unknown"
    assert result["aggregate"]["note"] is not None


def test_atomic_write_no_tmp_left_and_json_round_trips(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, [
        {"ticket_id": "T900", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": True},
    ])

    out_path = tmp_path / "out" / "leakage_audit.json"
    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    leakage_audit._persist_result(result, out_path)

    assert out_path.exists()
    tmp_marker = out_path.parent / f".{out_path.name}.tmp"
    assert not tmp_marker.exists()

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["audited_count"] == result["audited_count"]
    assert loaded["rows"][0]["ticket_id"] == "T900"


def test_aggregates_correct_on_small_synthetic_set(tmp_path):
    tickets = [
        _base_ticket(id="T900", category="easy"),
        _base_ticket(id="T901", category="easy"),
        _base_ticket(id="T902", category="tool_heavy"),
    ]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    per_answer = [
        # T900: leaked_strong
        {"ticket_id": "T900", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": True},
        # T901: resistant
        {"ticket_id": "T901", "tier": 1, "repeat": 0, "kind": "semantic_match", "verdict": False},
        # T902: no tier1 data -> unknown
    ]
    tc_path = tmp_path / "tier_compare.json"
    _write_tier_compare(tc_path, per_answer)

    result = leakage_audit.audit_tickets(tickets_path, tc_path)
    agg = result["aggregate"]
    assert agg["overall_counts"]["leaked_strong"] == 1
    assert agg["overall_counts"]["resistant"] == 1
    assert agg["overall_counts"]["unknown"] == 1
    assert agg["overall_counts"]["leaked_partial"] == 0
    assert agg["leak_rate"] == pytest.approx(1 / 3)
    assert agg["per_category_counts"]["easy"]["leaked_strong"] == 1
    assert agg["per_category_counts"]["easy"]["resistant"] == 1
    assert agg["per_category_counts"]["tool_heavy"]["unknown"] == 1
    # all three tickets have lexical overlap via the base ticket text/gold slug
    assert agg["n_with_lexical_overlap"] == 3


# ---------------------------------------------------------------------------
# config key + IRCA_TICKETS_PATH honoured by the CLI
# ---------------------------------------------------------------------------


def test_audit_tickets_config_carries_resolved_path_and_sha256(tmp_path):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "tickets.json"
    _write_tickets(tickets_path, tickets)

    result = leakage_audit.audit_tickets(tickets_path, tmp_path / "no_such_tier_compare.json")

    expected_sha = tickets_content_sha256(tickets_path)
    assert result["config"]["tickets_path"] == str(tickets_path)
    assert result["config"]["tickets_sha256"] == expected_sha


@pytest.fixture
def _clean_tickets_env():
    yield
    os.environ.pop("IRCA_TICKETS_PATH", None)


def test_main_honours_irca_tickets_path_env_var(tmp_path, _clean_tickets_env):
    ticket_only_in_temp_file = _base_ticket(id="T999", category="easy")
    tickets_path = tmp_path / "temp_tickets.json"
    _write_tickets(tickets_path, [ticket_only_in_temp_file])

    os.environ["IRCA_TICKETS_PATH"] = str(tickets_path)

    out_path = tmp_path / "leakage_audit_out.json"
    leakage_audit.main(["--tier-compare", str(tmp_path / "missing.json"), "--out", str(out_path)])

    result = json.loads(out_path.read_text(encoding="utf-8"))
    ids = {row["ticket_id"] for row in result["rows"]}
    assert ids == {"T999"}


def test_main_output_config_reflects_env_ticket_set(tmp_path, _clean_tickets_env):
    tickets = [_base_ticket(id="T900")]
    tickets_path = tmp_path / "temp_tickets.json"
    _write_tickets(tickets_path, tickets)

    os.environ["IRCA_TICKETS_PATH"] = str(tickets_path)

    out_path = tmp_path / "leakage_audit_out.json"
    leakage_audit.main(["--tier-compare", str(tmp_path / "missing.json"), "--out", str(out_path)])

    result = json.loads(out_path.read_text(encoding="utf-8"))
    expected_sha = tickets_content_sha256(tickets_path)
    assert result["config"]["tickets_path"] == str(tickets_path)
    assert result["config"]["tickets_sha256"] == expected_sha
