import json

import pytest

from eval import validate_tickets


def _ticket(
    ticket_id,
    ticket_text,
    category="easy",
    gold_root_cause="db_connection_pool_exhaustion",
    expected_behavior="resolve_with_approval",
    notes="some notes",
    gold_runbook_id="RB-DB-001.md",
    required_tools=None,
    min_confidence_evidence_sources=2,
):
    return {
        "id": ticket_id,
        "ticket_text": ticket_text,
        "category": category,
        "gold_root_cause": gold_root_cause,
        "gold_runbook_id": gold_runbook_id,
        "required_tools": required_tools or ["query_logs", "query_metrics"],
        "min_confidence_evidence_sources": min_confidence_evidence_sources,
        "expected_behavior": expected_behavior,
        "notes": notes,
    }


CLEAN_TEXT = "Checkout API returns 500s for 5% of requests since 10:00 AM; customers report failed payments."
LEAKY_TEXT = "Checkout API failing due to db connection pool exhaustion, connections pinned near max."
METRIC_LEAKY_TEXT = "db_pool_active_connections is pinned at 100 for the checkout service."
SHORT_TEXT = "Broke."
GENERIC_TEXT = "The service is broken and nothing works right now for some users."


def _write(path, tickets):
    path.write_text(json.dumps(tickets), encoding="utf-8")
    return path


def _stub_signature(monkeypatch, mapping):
    """mapping: {ticket_id: {"log_patterns": [...], "metric_name": "..."}}"""
    def fake_signature_for_ticket(ticket_id):
        return mapping.get(ticket_id)

    monkeypatch.setattr(validate_tickets.fake_data, "signature_for_ticket", fake_signature_for_ticket)


DEFAULT_SIGNATURE = {
    "log_patterns": ["{timestamp} {service}: too many connections open, request rejected"],
    "metric_name": "db_pool_active_connections",
}


def test_all_checks_pass_clean_ticket(tmp_path, monkeypatch):
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", CLEAN_TEXT)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})

    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    assert result["exit_ok"] is True
    assert result["fatal"] is False
    assert result["in_scope_count"] == 1

    exit_code = validate_tickets.main(["--baseline", str(base_path), "--candidate", str(cand_path)])
    assert exit_code == 0


@pytest.mark.parametrize("mutation", ["added", "removed", "reordered"])
def test_s1_fails_on_id_mismatch(tmp_path, monkeypatch, mutation):
    baseline = [_ticket("T001", CLEAN_TEXT), _ticket("T002", CLEAN_TEXT)]
    if mutation == "added":
        candidate = [_ticket("T001", CLEAN_TEXT), _ticket("T002", CLEAN_TEXT), _ticket("T003", CLEAN_TEXT)]
    elif mutation == "removed":
        candidate = [_ticket("T001", CLEAN_TEXT)]
    else:
        candidate = [_ticket("T002", CLEAN_TEXT), _ticket("T001", CLEAN_TEXT)]

    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE, "T002": DEFAULT_SIGNATURE, "T003": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    assert result["fatal"] is True
    assert result["structural_checks"][0]["passed"] is False
    assert result["exit_ok"] is False


@pytest.mark.parametrize("field,new_value", [
    ("gold_root_cause", "memory_cache_overgrowth"),
    ("expected_behavior", "escalate"),
    ("notes", "different notes"),
])
def test_s2_fails_on_non_text_field_change(tmp_path, monkeypatch, field, new_value):
    baseline = [_ticket("T001", CLEAN_TEXT)]
    changed = _ticket("T001", CLEAN_TEXT)
    changed[field] = new_value
    candidate = [changed]

    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    assert result["fatal"] is True
    s2 = result["structural_checks"][1]
    assert s2["passed"] is False
    assert any(field in line and "T001" in line for line in s2["detail"])


def test_s3_fails_on_rewritten_ambiguous_ticket(tmp_path, monkeypatch):
    baseline = [
        _ticket("T001", CLEAN_TEXT),
        _ticket("T002", "ambiguous text here", category="ambiguous", gold_root_cause=None, expected_behavior="escalate"),
    ]
    candidate = [
        _ticket("T001", CLEAN_TEXT),
        _ticket("T002", "rewritten ambiguous text", category="ambiguous", gold_root_cause=None, expected_behavior="escalate"),
    ]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE, "T002": None})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    assert result["fatal"] is True
    s3 = result["structural_checks"][2]
    assert s3["passed"] is False
    assert any("T002" in line for line in s3["detail"])


def test_s3_fails_on_rewritten_adversarial_ticket(tmp_path, monkeypatch):
    baseline = [
        _ticket("T001", CLEAN_TEXT),
        _ticket("T002", "adversarial text here", category="adversarial", gold_root_cause=None, expected_behavior="escalate"),
    ]
    candidate = [
        _ticket("T001", CLEAN_TEXT),
        _ticket("T002", "rewritten adversarial text", category="adversarial", gold_root_cause=None, expected_behavior="escalate"),
    ]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE, "T002": None})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    assert result["fatal"] is True
    s3 = result["structural_checks"][2]
    assert s3["passed"] is False
    assert any("T002" in line for line in s3["detail"])


def test_l1_fails_on_gold_token_including_inflection(tmp_path, monkeypatch):
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", LEAKY_TEXT)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    l1 = result["lexical_checks"][0]
    assert l1["check"] == "L1"
    assert l1["passed"] is False
    # "connections" should be caught as inflected form of gold token "connection"
    assert any("connection" in line.lower() for line in l1["detail"])
    assert any("T001" in line for line in l1["detail"])
    assert result["exit_ok"] is False


def test_l2_fails_on_metric_token(tmp_path, monkeypatch):
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", METRIC_LEAKY_TEXT)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    l2 = result["lexical_checks"][1]
    assert l2["check"] == "L2"
    assert l2["passed"] is False
    assert any("T001" in line for line in l2["detail"])
    assert result["exit_ok"] is False


def test_d1_pre_existing_when_both_baseline_and_candidate_fail(tmp_path, monkeypatch):
    # D1 reads only the fake_data signature, which is identical for baseline
    # and candidate (only ticket_text differs), so a not-derivable signature
    # always fails on both sides -- this must be reported as pre_existing,
    # not a regression, and must not fail the exit code.
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", CLEAN_TEXT)]
    no_signal_signature = {
        "log_patterns": ["{timestamp} {service}: something unrelated happened"],
        "metric_name": "unrelated_metric_name",
    }
    _stub_signature(monkeypatch, {"T001": no_signal_signature})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    (d1,) = result["derivability_checks"]
    assert d1["check"] == "D1"
    assert d1["passed"] is True
    assert d1["regressions"] == 0
    assert d1["pre_existing"] == 1
    assert any("T001" in line for line in d1["pre_existing_detail"])
    assert result["exit_ok"] is True

    exit_code = validate_tickets.main(["--baseline", str(base_path), "--candidate", str(cand_path)])
    assert exit_code == 0


def test_d1_regression_when_candidate_fails_and_baseline_passed(monkeypatch):
    # Call _check_d1 directly (rather than through validate()) so this test
    # is not order-coupled to how many other checks call
    # fake_data.signature_for_ticket first. Simulates a hypothetical future
    # change to fake_data's signatures: same ticket id, baseline WAS
    # derivable, candidate is not -- the only case D1 should FAIL.
    candidate_ticket = _ticket("T001", CLEAN_TEXT)
    baseline_ticket = _ticket("T001", CLEAN_TEXT)

    def stub_ticket_derivable(ticket):
        return ticket is baseline_ticket

    monkeypatch.setattr(validate_tickets, "_ticket_derivable", stub_ticket_derivable)
    monkeypatch.setattr(validate_tickets, "_signature_text", lambda ticket_id: ([], "stub"))

    result = validate_tickets._check_d1([candidate_ticket], {"T001": baseline_ticket})
    assert result["check"] == "D1"
    assert result["passed"] is False
    assert result["regressions"] == 1
    assert result["pre_existing"] == 0
    assert any("T001" in line for line in result["regression_detail"])


def test_w1_and_w2_warn_not_fail(tmp_path, monkeypatch):
    baseline = [_ticket("T001", CLEAN_TEXT), _ticket("T002", CLEAN_TEXT)]
    candidate = [_ticket("T001", SHORT_TEXT), _ticket("T002", GENERIC_TEXT)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE, "T002": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    w1, w2 = result["warnings"]
    assert w1["check"] == "W1"
    assert w1["warned"] is True
    assert any("T001" in line for line in w1["detail"])

    assert w2["check"] == "W2"
    # baseline CLEAN_TEXT had an observable, both candidate texts (SHORT_TEXT,
    # GENERIC_TEXT) do not -- the rewrite removed it for both, a regression.
    assert w2["warned"] is True
    assert w2["regressions"] == 2
    assert any("T001" in line for line in w2["regression_detail"])
    assert any("T002" in line for line in w2["regression_detail"])

    # L1/L2/D1 should still pass since these tickets have no leakage/derivability issues
    l1, l2, l3 = result["lexical_checks"]
    (d1,) = result["derivability_checks"]
    assert l1["passed"] and l2["passed"] and l3["passed"] and d1["passed"]

    assert result["exit_ok"] is True

    exit_code = validate_tickets.main(["--baseline", str(base_path), "--candidate", str(cand_path)])
    assert exit_code == 0


def test_w2_pre_existing_when_neither_baseline_nor_candidate_has_observable(tmp_path, monkeypatch):
    baseline = [_ticket("T001", GENERIC_TEXT)]
    candidate = [_ticket("T001", GENERIC_TEXT)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    _, w2 = result["warnings"]
    assert w2["check"] == "W2"
    assert w2["warned"] is False
    assert w2["regressions"] == 0
    assert w2["pre_existing"] == 1
    assert any("T001" in line for line in w2["pre_existing_detail"])
    assert result["exit_ok"] is True

    exit_code = validate_tickets.main(["--baseline", str(base_path), "--candidate", str(cand_path)])
    assert exit_code == 0


def test_clean_in_scope_ticket_passes_l1_l2_d1(tmp_path, monkeypatch):
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", CLEAN_TEXT)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    l1, l2, l3 = result["lexical_checks"]
    (d1,) = result["derivability_checks"]
    assert l1["passed"] is True
    assert l2["passed"] is True
    assert l3["passed"] is True
    assert d1["passed"] is True
    assert result["exit_ok"] is True


@pytest.mark.parametrize("text,expected_term", [
    ("The dashboard shows the checkout error rate climbing since 10am.", "dashboard"),
    ("app logs show connection failures for the checkout service.", "log"),
    ("the ingress queue is at 85% capacity per the metrics.", "metric"),
    ("Support says the alert fired twice overnight for checkout.", "alert"),
    ("checkout monitors have been flapping since this morning.", "monitor"),
])
def test_l3_fails_on_telemetry_reference(tmp_path, monkeypatch, text, expected_term):
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", text)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    l1, l2, l3 = result["lexical_checks"]
    assert l3["check"] == "L3"
    assert l3["passed"] is False
    assert any(expected_term in line.lower() and "T001" in line for line in l3["detail"])
    assert result["exit_ok"] is False

    exit_code = validate_tickets.main(["--baseline", str(base_path), "--candidate", str(cand_path)])
    assert exit_code == 1


def test_l3_whole_word_matching_does_not_match_substrings(tmp_path, monkeypatch):
    text = "The blogs and catalog for checkout monitors were reviewed since 10am."
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", text)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    _, _, l3 = result["lexical_checks"]
    assert l3["passed"] is False
    # "blogs" and "catalog" must not be flagged as matches
    assert not any("'blogs'" in line or "matched 'blogs'" in line for line in l3["detail"])
    assert not any("'catalog'" in line or "matched 'catalog'" in line for line in l3["detail"])
    # "monitors" (whole word, plural of monitor) IS a legitimate match
    assert any("monitor" in line.lower() and "T001" in line for line in l3["detail"])


def test_l3_passes_on_clean_complaint_style_text(tmp_path, monkeypatch):
    clean_complaint = "Checkout is returning 500s for about 5% of users since 10am."
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", clean_complaint)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    _, _, l3 = result["lexical_checks"]
    assert l3["check"] == "L3"
    assert l3["passed"] is True
    assert l3["detail"] == []
    assert result["exit_ok"] is True


def test_l3_out_of_scope_tickets_not_checked(tmp_path, monkeypatch):
    telemetry_text = "the dashboard shows the alert firing repeatedly per the logs."
    baseline = [
        _ticket("T001", CLEAN_TEXT),
        _ticket("T002", "ambiguous text here", category="ambiguous", gold_root_cause=None, expected_behavior="escalate"),
        _ticket("T003", "adversarial text here", category="adversarial", gold_root_cause=None, expected_behavior="escalate"),
    ]
    candidate = [
        _ticket("T001", CLEAN_TEXT),
        _ticket("T002", telemetry_text, category="ambiguous", gold_root_cause=None, expected_behavior="escalate"),
        _ticket("T003", telemetry_text, category="adversarial", gold_root_cause=None, expected_behavior="escalate"),
    ]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE, "T002": None, "T003": None})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    # S3 fatally fails first since T002/T003 (out-of-scope) were rewritten,
    # so per-ticket checks never run -- confirm L3 specifically by calling
    # the check function directly on an in-scope-filtered list instead.
    in_scope = [t for t in candidate if not validate_tickets._out_of_scope(t)]
    assert [t["id"] for t in in_scope] == ["T001"]
    l3 = validate_tickets._check_l3(in_scope)
    assert l3["passed"] is True


def test_exit_code_1_when_only_l3_fails(tmp_path, monkeypatch):
    telemetry_text = "the dashboard shows checkout failing for 5% of users since 10am."
    baseline = [_ticket("T001", CLEAN_TEXT)]
    candidate = [_ticket("T001", telemetry_text)]
    _stub_signature(monkeypatch, {"T001": DEFAULT_SIGNATURE})
    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    result = validate_tickets.validate(base_path, cand_path)
    l1, l2, l3 = result["lexical_checks"]
    (d1,) = result["derivability_checks"]
    assert l1["passed"] is True
    assert l2["passed"] is True
    assert l3["passed"] is False
    assert d1["passed"] is True

    exit_code = validate_tickets.main(["--baseline", str(base_path), "--candidate", str(cand_path)])
    assert exit_code == 1
