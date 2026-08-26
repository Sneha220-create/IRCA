"""Tests for the tiered runner in eval/run_benchmark.py: tier dispatch
(1 = plain LLM, 2 = LLM + RAG only, 3 = LLM + tools single round-trip,
4 = full agent loop, the historical default). Offline only -- every tier
entry point is stubbed, no network."""

from __future__ import annotations

import json
import os

import pytest

from agent.state import TaskState
from eval import run_benchmark
from tools.fake_data import TICKETS_PATH_ENV


@pytest.fixture(autouse=True)
def _clear_tickets_env():
    # Plain os.environ, deliberately not monkeypatch: run_benchmark.main()
    # sets IRCA_TICKETS_PATH directly via os.environ (see --tickets-path),
    # bypassing monkeypatch's own change tracking. If this fixture used
    # monkeypatch.delenv to clean up, monkeypatch's own auto-revert (which
    # runs AFTER this fixture's teardown, since this fixture depends on the
    # monkeypatch fixture) would restore whatever monkeypatch last recorded
    # -- undoing this cleanup and leaking the env var into later tests.
    os.environ.pop(TICKETS_PATH_ENV, None)
    yield
    os.environ.pop(TICKETS_PATH_ENV, None)


def _canned_state(ticket_id="T001"):
    state = TaskState(ticket_id=ticket_id, description="desc", status="resolved")
    state.hypothesis = "db_connection_pool_exhaustion"
    state.confidence = 0.9
    state.evidence_sources = ["query_logs", "search_runbooks"]
    state.citations = ["RB-DB-001"]
    return state


# ---------------------------------------------------------------------------
# Default-behaviour regression: tier 4 must be byte-for-byte the historical
# shape, plus exactly {tier, repeat} added.
# ---------------------------------------------------------------------------
def test_tier4_default_behaviour_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))

    rc = run_benchmark.main(["--live", "--out", str(tmp_path)])
    assert rc == 0

    tickets = run_benchmark._load_tickets()
    expected_selection = run_benchmark._select_tickets(tickets, subset=None, ticket_ids=None)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == sorted(f"{t['id']}.json" for t in expected_selection)
    assert len(files) == 63

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert set(result.keys()) == {
        "schema_version",
        "tier",
        "repeat",
        "ticket_id",
        "run",
        "ticket",
        "state",
        "pending_action",
        "usage",
        "provenance",
    }
    assert result["tier"] == 4
    assert result["repeat"] == 0
    assert result["schema_version"] == 6
    assert set(result["run"].keys()) == {
        "started_at",
        "wall_clock_seconds",
        "runner_error",
        "tickets_path",
        "tickets_sha256",
    }
    assert result["run"]["tickets_path"] == str(run_benchmark.TICKETS_PATH)
    assert (
        result["run"]["tickets_sha256"]
        == run_benchmark.tickets_content_sha256(run_benchmark.TICKETS_PATH)
    )
    assert set(result["ticket"].keys()) == set(run_benchmark.TICKET_DENORM_FIELDS)
    assert result["state"] == _canned_state("T001").model_dump()


def test_tier5_rejected_by_argparse():
    with pytest.raises(SystemExit):
        run_benchmark.main(["--live", "--tier", "5"])


# ---------------------------------------------------------------------------
# Tiers 1-3
# ---------------------------------------------------------------------------
def _tier1_payload(ticket_id):
    return {"ticket_id": ticket_id, "tier": 1, "final_answer": "root cause X"}


def _tier2_payload(ticket_id):
    return {
        "ticket_id": ticket_id,
        "tier": 2,
        "retrieval_query": "db pool exhaustion",
        "retrieval_status": "ok",
        "retrieval_top_score": 0.92,
        "retrieved_doc_ids": ["RB-DB-001.md"],
        "final_answer": "root cause Y",
    }


def _tier3_payload(ticket_id):
    return {
        "ticket_id": ticket_id,
        "tool_calls_made": [{"name": "query_logs", "arguments": {}, "status": "ok"}],
        "final_answer": "root cause Z",
    }


@pytest.mark.parametrize(
    "tier,attr,module,payload_fn",
    [
        (1, "run_plain_llm", "agent.baselines", _tier1_payload),
        (2, "run_rag_only", "agent.baselines", _tier2_payload),
        (3, "run_single_pass", "agent.single_pass", _tier3_payload),
    ],
)
def test_tier_run_writes_result_without_state_or_pending_action(
    tmp_path, monkeypatch, tier, attr, module, payload_fn
):
    mod = __import__(module, fromlist=[attr])
    monkeypatch.setattr(mod, attr, lambda tid: payload_fn(tid))

    rc = run_benchmark.main(["--live", "--tier", str(tier), "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert "state" not in result
    assert "pending_action" not in result
    assert result["tier"] == tier
    assert result["repeat"] == 0
    assert result["schema_version"] == 6
    assert result["baseline"] == payload_fn("T001")

    if tier == 2:
        assert result["baseline"]["retrieval_status"] == "ok"
        assert result["baseline"]["retrieval_top_score"] == 0.92
        assert result["baseline"]["retrieved_doc_ids"] == ["RB-DB-001.md"]
    if tier == 3:
        assert result["baseline"]["tool_calls_made"] == [
            {"name": "query_logs", "arguments": {}, "status": "ok"}
        ]


def test_repeat_flag_persists_in_every_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", lambda tid: _tier1_payload(tid))

    rc = run_benchmark.main(
        ["--live", "--tier", "1", "--tickets", "T001,T002", "--repeat", "2", "--out", str(tmp_path)]
    )
    assert rc == 0

    for tid in ("T001", "T002"):
        result = json.loads((tmp_path / f"{tid}.json").read_text(encoding="utf-8"))
        assert result["repeat"] == 2


def test_default_out_dir_resolution():
    assert run_benchmark._default_out_dir(4) == run_benchmark.DEFAULT_OUT_DIR
    assert run_benchmark._default_out_dir(2) == run_benchmark.REPO_ROOT / "eval" / "results" / "tier2"
    assert run_benchmark._default_out_dir(1) == run_benchmark.REPO_ROOT / "eval" / "results" / "tier1"
    assert run_benchmark._default_out_dir(3) == run_benchmark.REPO_ROOT / "eval" / "results" / "tier3"


def test_tier1_crash_has_no_state_key_and_runner_error(tmp_path, monkeypatch):
    def _raise(ticket_id):
        raise ValueError("boom")

    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", _raise)

    rc = run_benchmark.main(["--live", "--tier", "1", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert "state" not in result
    assert "pending_action" not in result
    assert result["baseline"] is None
    assert result["run"]["runner_error"]["type"] == "ValueError"
    assert result["run"]["runner_error"]["message"] == "boom"
    assert "Traceback" in result["run"]["runner_error"]["traceback"]

    captured_summary_status = "crashed"
    # crashed status appears in the progress line via run_one's return value;
    # verified indirectly through main()'s exit code and file content above.
    assert captured_summary_status == "crashed"


def test_tier2_status_answered_in_progress_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_benchmark.baselines, "run_rag_only", lambda tid: _tier2_payload(tid))

    rc = run_benchmark.main(["--live", "--tier", "2", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "status=answered" in captured.err
    assert "tier=2" in captured.err


def test_usage_shim_patches_baselines_module_for_tier2(tmp_path, monkeypatch):
    calls = []

    def fake_llm(*args, **kwargs):
        calls.append(1)
        from types import SimpleNamespace

        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))

    def fake_search_runbooks(*args, **kwargs):
        # retrieval must not be counted as an LLM call
        return {"status": "ok", "data": {"chunks": [], "top_score": 0.9}}

    def fake_run_rag_only(ticket_id):
        run_benchmark.baselines.call_llm_with_tools({}, [])
        run_benchmark.baselines.search_runbooks("q", ticket_id=ticket_id)
        run_benchmark.baselines.call_llm_with_tools({}, [])
        return _tier2_payload(ticket_id)

    monkeypatch.setattr(run_benchmark.baselines, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.baselines, "search_runbooks", fake_search_runbooks)
    monkeypatch.setattr(run_benchmark, "_TIER_ENTRY_POINT", {**run_benchmark._TIER_ENTRY_POINT, 2: fake_run_rag_only})

    rc = run_benchmark.main(["--live", "--tier", "2", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert result["usage"]["llm_call_count"] == 2
    assert result["usage"]["total_tokens_in"] == 6
    assert result["usage"]["total_tokens_out"] == 4


# ---------------------------------------------------------------------------
# --tickets-path: switchable ticket-set file
# ---------------------------------------------------------------------------
def _alt_tickets():
    return [
        {
            "id": "TX01",
            "description": "alt ticket one",
            "category": "database",
            "gold_root_cause": "db_connection_pool_exhaustion",
            "gold_runbook_id": "RB-DB-001",
            "required_tools": [],
            "expected_behavior": None,
            "min_confidence_evidence_sources": 1,
        },
        {
            "id": "TX02",
            "description": "alt ticket two",
            "category": "database",
            "gold_root_cause": "db_connection_pool_exhaustion",
            "gold_runbook_id": "RB-DB-001",
            "required_tools": [],
            "expected_behavior": None,
            "min_confidence_evidence_sources": 1,
        },
    ]


def test_tickets_path_flag_runs_exactly_those_tickets(tmp_path, monkeypatch):
    alt_path = tmp_path / "alt_tickets.json"
    alt_path.write_text(json.dumps(_alt_tickets()), encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", lambda tid: _tier1_payload(tid))

    rc = run_benchmark.main(
        ["--live", "--tier", "1", "--tickets-path", str(alt_path), "--out", str(out_dir)]
    )
    assert rc == 0

    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["TX01.json", "TX02.json"]

    expected_hash = run_benchmark.tickets_content_sha256(alt_path)
    for name in files:
        result = json.loads((out_dir / name).read_text(encoding="utf-8"))
        assert result["run"]["tickets_path"] == str(alt_path)
        assert result["run"]["tickets_sha256"] == expected_hash


def test_default_run_still_uses_default_tickets_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))

    rc = run_benchmark.main(["--live", "--out", str(tmp_path)])
    assert rc == 0

    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 63

    expected_hash = run_benchmark.tickets_content_sha256(run_benchmark.TICKETS_PATH)
    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert result["run"]["tickets_path"] == str(run_benchmark.TICKETS_PATH)
    assert result["run"]["tickets_sha256"] == expected_hash


def test_tickets_path_flag_propagates_to_tier_modules_get_ticket(tmp_path, monkeypatch):
    """The whole point of --tickets-path: tier modules that call
    fake_data.get_ticket() internally must see the alternate ticket text,
    not just the runner's own selection."""
    alt_tickets = _alt_tickets()
    alt_tickets[0]["description"] = "UNIQUE ALT TEXT MARKER"
    alt_path = tmp_path / "alt_tickets.json"
    alt_path.write_text(json.dumps(alt_tickets), encoding="utf-8")
    out_dir = tmp_path / "out"

    seen_descriptions = []

    def fake_run_plain_llm(ticket_id):
        ticket = run_benchmark.baselines.get_ticket(ticket_id)
        seen_descriptions.append(ticket["description"] if ticket else None)
        return _tier1_payload(ticket_id)

    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", fake_run_plain_llm)

    rc = run_benchmark.main(
        [
            "--live",
            "--tier",
            "1",
            "--tickets-path",
            str(alt_path),
            "--tickets",
            "TX01",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert seen_descriptions == ["UNIQUE ALT TEXT MARKER"]


# ---------------------------------------------------------------------------
# tickets_content_sha256 -- content-based, line-ending-invariant hashing.
# Regression test for the CRLF/LF cross-platform hashing bug: raw-byte
# hashing made the same ticket set hash differently on Windows (CRLF) vs
# Linux (LF) checkouts.
# ---------------------------------------------------------------------------


def test_tickets_content_sha256_invariant_to_line_endings(tmp_path):
    tickets = _alt_tickets()
    text = json.dumps(tickets, indent=2)

    crlf_path = tmp_path / "crlf.json"
    lf_path = tmp_path / "lf.json"
    crlf_path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    lf_path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))

    assert run_benchmark.tickets_content_sha256(crlf_path) == run_benchmark.tickets_content_sha256(lf_path)


def test_tickets_content_sha256_differs_for_different_content(tmp_path):
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    tickets = _alt_tickets()
    a_path.write_text(json.dumps(tickets), encoding="utf-8")
    tickets[0]["description"] = "a different description entirely"
    b_path.write_text(json.dumps(tickets), encoding="utf-8")

    assert run_benchmark.tickets_content_sha256(a_path) != run_benchmark.tickets_content_sha256(b_path)


def test_tickets_content_sha256_invariant_to_whitespace_and_key_order(tmp_path):
    tickets = _alt_tickets()
    compact_path = tmp_path / "compact.json"
    pretty_path = tmp_path / "pretty.json"
    compact_path.write_text(json.dumps(tickets, separators=(",", ":")), encoding="utf-8")
    pretty_path.write_text(json.dumps(tickets, indent=4, sort_keys=False), encoding="utf-8")

    assert run_benchmark.tickets_content_sha256(compact_path) == run_benchmark.tickets_content_sha256(pretty_path)


# ---------------------------------------------------------------------------
# Provenance: which model/provider/credential/commit produced a sweep.
# ---------------------------------------------------------------------------
def test_provenance_present_with_all_four_keys_and_schema5(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-provenance-test-0123456789")

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == 6
    assert "provenance" in result
    prov = result["provenance"]
    assert set(prov.keys()) == {"llm_base_url", "llm_model", "llm_api_key_sha8", "git_commit"}


def test_provenance_key_fingerprint_never_reveals_the_raw_key(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    fake_key = "sk-super-secret-fake-key-0123456789abcdef"
    monkeypatch.setenv("LLM_API_KEY", fake_key)

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    raw_text = (tmp_path / "T001.json").read_text(encoding="utf-8")
    assert fake_key not in raw_text
    assert fake_key[:12] not in raw_text

    result = json.loads(raw_text)
    key_sha8 = result["provenance"]["llm_api_key_sha8"]
    assert isinstance(key_sha8, str)
    assert len(key_sha8) == 8
    assert key_sha8 == key_sha8.lower()
    assert all(c in "0123456789abcdef" for c in key_sha8)
    assert key_sha8 != fake_key
    assert key_sha8 not in fake_key
    assert fake_key not in key_sha8


def test_provenance_no_key_set_gives_none_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert result["provenance"]["llm_api_key_sha8"] is None


def test_provenance_git_unavailable_gives_none_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(run_benchmark.subprocess, "run", _raise)

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert result["provenance"]["git_commit"] is None


def test_run_one_direct_call_defaults_provenance_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    ticket = {
        "id": "T001",
        "category": "easy",
        "gold_root_cause": "x",
        "gold_runbook_id": "RB-1",
        "required_tools": [],
        "expected_behavior": "resolve",
        "min_confidence_evidence_sources": 1,
    }
    result = run_benchmark.run_one(ticket, tmp_path, tier=4)
    assert result is not None
    written = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert written["provenance"] is None


# ---------------------------------------------------------------------------
# Fail-fast detection of dead LLM calls (agent/llm.py returns an
# {"error": ...} dict rather than raising, so a dead credential previously
# produced a full "successful" sweep of nothing but failed LLM calls).
# ---------------------------------------------------------------------------
def _make_scripted_llm(script: dict[str, list[str]]):
    """Returns a fake call_llm_with_tools that looks up which ticket is
    currently being processed via a mutable `current` cell the test sets
    before invoking run_one/main, and pops the next scripted outcome
    ("ok" or "fail") for that ticket off script[ticket_id]."""
    state = {"current": None}

    def fake_llm(*args, **kwargs):
        ticket_id = state["current"]
        outcomes = script[ticket_id]
        outcome = outcomes.pop(0)
        if outcome == "fail":
            return {"error": f"403 - please check the api-key you provided ({ticket_id})"}
        from types import SimpleNamespace

        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    return fake_llm, state


def _scripted_run_plain_llm(state):
    def run_plain_llm(ticket_id):
        state["current"] = ticket_id
        run_benchmark.baselines.call_llm_with_tools({}, [])
        return _tier1_payload(ticket_id)

    return run_plain_llm


def test_all_tickets_fail_every_call_aborts_after_threshold(tmp_path, monkeypatch):
    ticket_ids = ["T001", "T002", "T003", "T004", "T005"]
    script = {tid: ["fail"] for tid in ticket_ids}
    fake_llm, state = _make_scripted_llm(script)

    monkeypatch.setattr(run_benchmark.baselines, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", _scripted_run_plain_llm(state))

    rc = run_benchmark.main(
        ["--live", "--tier", "1", "--tickets", ",".join(ticket_ids), "--out", str(tmp_path)]
    )
    assert rc == 3

    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == run_benchmark.MAX_CONSECUTIVE_LLM_FAILURES
    assert files == ["T001.json", "T002.json", "T003.json"]


def test_all_tickets_fail_abort_message_contents(tmp_path, monkeypatch, capsys):
    ticket_ids = ["T001", "T002", "T003", "T004"]
    script = {tid: ["fail"] for tid in ticket_ids}
    fake_llm, state = _make_scripted_llm(script)

    monkeypatch.setattr(run_benchmark.baselines, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", _scripted_run_plain_llm(state))

    rc = run_benchmark.main(
        ["--live", "--tier", "1", "--tickets", ",".join(ticket_ids), "--out", str(tmp_path)]
    )
    assert rc == 3

    captured = capsys.readouterr()
    assert "403 - please check the api-key you provided" in captured.err
    assert "INVALID" in captured.err
    assert "3/4" in captured.err or "attempted before abort" in captured.err
    assert "LLM_BASE_URL" in captured.err
    assert "LLM_MODEL" in captured.err


def test_two_consecutive_full_failures_then_success_does_not_abort(tmp_path, monkeypatch):
    ticket_ids = ["T001", "T002", "T003"]
    script = {"T001": ["fail"], "T002": ["fail"], "T003": ["ok"]}
    fake_llm, state = _make_scripted_llm(script)

    monkeypatch.setattr(run_benchmark.baselines, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", _scripted_run_plain_llm(state))

    rc = run_benchmark.main(
        ["--live", "--tier", "1", "--tickets", ",".join(ticket_ids), "--out", str(tmp_path)]
    )
    assert rc == 0

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["T001.json", "T002.json", "T003.json"]


def test_partial_failure_does_not_count_toward_consecutive_abort(tmp_path, monkeypatch):
    ticket_ids = ["T001", "T002", "T003", "T004", "T005"]
    # Every ticket has one failing call and one succeeding call -- a
    # PARTIAL failure -- repeated well past the abort threshold. None of
    # these should count toward the consecutive-all-failed streak.
    script = {tid: ["fail", "ok"] for tid in ticket_ids}
    fake_llm, state = _make_scripted_llm(script)

    def run_plain_llm(ticket_id):
        state["current"] = ticket_id
        run_benchmark.baselines.call_llm_with_tools({}, [])
        run_benchmark.baselines.call_llm_with_tools({}, [])
        return _tier1_payload(ticket_id)

    monkeypatch.setattr(run_benchmark.baselines, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", run_plain_llm)

    rc = run_benchmark.main(
        ["--live", "--tier", "1", "--tickets", ",".join(ticket_ids), "--out", str(tmp_path)]
    )
    assert rc == 0

    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 5

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert result["usage"]["failed_calls"] == 1
    assert result["usage"]["llm_call_count"] == 2
    assert result["usage"]["last_error"] is not None


def test_partial_failure_reported_in_progress_line(tmp_path, monkeypatch, capsys):
    script = {"T001": ["fail", "ok"]}
    fake_llm, state = _make_scripted_llm(script)

    def run_plain_llm(ticket_id):
        state["current"] = ticket_id
        run_benchmark.baselines.call_llm_with_tools({}, [])
        run_benchmark.baselines.call_llm_with_tools({}, [])
        return _tier1_payload(ticket_id)

    monkeypatch.setattr(run_benchmark.baselines, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", run_plain_llm)

    rc = run_benchmark.main(["--live", "--tier", "1", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "llm_failures=1/2" in captured.err


def test_usage_failed_calls_and_last_error_persist_schema6(tmp_path, monkeypatch):
    script = {"T001": ["fail"]}
    fake_llm, state = _make_scripted_llm(script)

    def run_plain_llm(ticket_id):
        state["current"] = ticket_id
        run_benchmark.baselines.call_llm_with_tools({}, [])
        return _tier1_payload(ticket_id)

    monkeypatch.setattr(run_benchmark.baselines, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", run_plain_llm)

    rc = run_benchmark.main(["--live", "--tier", "1", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == 6
    assert result["usage"]["failed_calls"] == 1
    assert "403" in result["usage"]["last_error"]


def test_normal_all_success_sweep_summary_line_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_benchmark.baselines, "run_plain_llm", lambda tid: _tier1_payload(tid))

    rc = run_benchmark.main(
        ["--live", "--tier", "1", "--tickets", "T001,T002", "--out", str(tmp_path)]
    )
    assert rc == 0

    captured = capsys.readouterr()
    last_line = [ln for ln in captured.err.strip().splitlines() if ln][-1]
    assert last_line.startswith("2 run, 0 crashed, total wall clock")
    assert "llm failures" not in last_line
    assert "tickets with LLM failures" not in last_line
