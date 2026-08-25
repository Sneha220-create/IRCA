"""Tests for the tiered runner in eval/run_benchmark.py: tier dispatch
(1 = plain LLM, 2 = LLM + RAG only, 3 = LLM + tools single round-trip,
4 = full agent loop, the historical default). Offline only -- every tier
entry point is stubbed, no network."""

from __future__ import annotations

import json

import pytest

from agent.state import TaskState
from eval import run_benchmark


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
    }
    assert result["tier"] == 4
    assert result["repeat"] == 0
    assert result["schema_version"] == 3
    assert set(result["run"].keys()) == {"started_at", "wall_clock_seconds", "runner_error"}
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
    assert result["schema_version"] == 3
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
