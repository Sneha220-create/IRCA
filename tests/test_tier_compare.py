import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eval import llm_judge, tier_compare
from eval.run_benchmark import tickets_content_sha256


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TICKET_A = {
    "id": "T001",
    "category": "easy",
    "ticket_text": "Checkout API returning 500s under load.",
    "gold_root_cause": "db_connection_pool_exhaustion",
    "expected_behavior": "resolve_with_approval",
}

TICKET_AMBIG = {
    "id": "T900",
    "category": "easy",
    "ticket_text": "Intermittent errors, cause unclear.",
    "gold_root_cause": "unknown",
    "expected_behavior": "escalate",
}


def _baseline_result(tier, ticket_id, answer, repeat=0):
    return {
        "tier": tier,
        "repeat": repeat,
        "ticket_id": ticket_id,
        "run": {"wall_clock_seconds": 1.0},
        "usage": {"llm_call_count": 1, "total_tokens_in": 10, "total_tokens_out": 10},
        "baseline": {"final_answer": answer} if answer is not None else None,
    }


def _tier4_result(ticket_id, hypothesis, status="resolved", repeat=0):
    return {
        "tier": 4,
        "repeat": repeat,
        "ticket_id": ticket_id,
        "run": {"wall_clock_seconds": 2.0},
        "usage": {"llm_call_count": 3, "total_tokens_in": 30, "total_tokens_out": 30},
        "state": {
            "ticket_id": ticket_id,
            "hypothesis": hypothesis,
            "status": status,
            "iteration": 0,
            "trajectory": [],
            "citations": [],
            "pending_action_id": None,
        }
        if hypothesis is not None
        else None,
    }


# ---------------------------------------------------------------------------
# candidate_answer
# ---------------------------------------------------------------------------


def test_candidate_answer_tiers_1_3():
    for tier in (1, 2, 3):
        r = _baseline_result(tier, "T001", "disk is full")
        assert tier_compare.candidate_answer(r) == "disk is full"


def test_candidate_answer_tier4():
    r = _tier4_result("T001", "disk is full")
    assert tier_compare.candidate_answer(r) == "disk is full"


def test_candidate_answer_crashed_baseline():
    r = _baseline_result(1, "T001", None)
    r["baseline"] = None
    assert tier_compare.candidate_answer(r) is None


def test_candidate_answer_crashed_tier4():
    r = _tier4_result("T001", None)
    assert tier_compare.candidate_answer(r) is None


def test_candidate_answer_tier4_never_falls_back_to_baseline():
    r = _tier4_result("T001", None)
    r["baseline"] = {"final_answer": "should never be read"}
    assert tier_compare.candidate_answer(r) is None


def test_candidate_answer_tiers_1_3_never_read_state():
    r = _baseline_result(1, "T001", None)
    r["baseline"] = None
    r["state"] = {"hypothesis": "should never be read"}
    assert tier_compare.candidate_answer(r) is None


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_interval_known_values():
    lo, hi = tier_compare.wilson_interval(0, 0)
    assert lo is None and hi is None

    # 1/1 success -> wide interval, but lower bound clearly > 0
    lo, hi = tier_compare.wilson_interval(1, 1)
    assert 0.2 < lo < 0.3
    assert hi == pytest.approx(1.0, abs=1e-9)

    # 50/100 -> approx symmetric around 0.5
    lo, hi = tier_compare.wilson_interval(50, 100)
    assert lo == pytest.approx(0.404, abs=0.01)
    assert hi == pytest.approx(0.596, abs=0.01)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def test_compare_to_tier4_tier4_ahead():
    cell4 = tier_compare.aggregate_cell([(9, 10), (9, 10)])
    other = tier_compare.aggregate_cell([(2, 10), (3, 10)])
    result = tier_compare.compare_to_tier4(cell4, other)
    assert result["verdict"] == "tier4_ahead"


def test_compare_to_tier4_other_ahead():
    cell4 = tier_compare.aggregate_cell([(2, 10), (3, 10)])
    other = tier_compare.aggregate_cell([(9, 10), (9, 10)])
    result = tier_compare.compare_to_tier4(cell4, other)
    assert result["verdict"] == "other_ahead"
    assert "other" in result["detail"] or "tier4" in result["detail"]


def test_compare_to_tier4_inconclusive_overlap():
    cell4 = tier_compare.aggregate_cell([(5, 10), (6, 10)])
    other = tier_compare.aggregate_cell([(5, 10), (5, 10)])
    result = tier_compare.compare_to_tier4(cell4, other)
    assert result["verdict"] == "inconclusive"
    assert "more repeats" in result["detail"]


# ---------------------------------------------------------------------------
# Aggregation across repeats
# ---------------------------------------------------------------------------


def test_aggregate_cell_across_3_repeats():
    cell = tier_compare.aggregate_cell([(1, 2), (2, 2), (0, 2)])
    assert cell["n_repeats"] == 3
    assert cell["per_repeat"] == [0.5, 1.0, 0.0]
    assert cell["median"] == 0.5
    assert cell["min"] == 0.0
    assert cell["max"] == 1.0
    assert cell["n_success_total"] == 3
    assert cell["n_trials_total"] == 6


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------


def test_blinding_prompts_no_tier_markers_and_stable(tmp_path):
    captured_semantic_prompts = []
    captured_semantic_args = []

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        prompt = llm_judge.build_prompt_3a(ticket_text, category, hypothesis, gold_root_cause)
        captured_semantic_prompts.append(prompt)
        captured_semantic_args.append((ticket_text, category, hypothesis, gold_root_cause))
        return True, "looks right", None

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier4").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_baseline_result(1, "T001", "the disk is full")), encoding="utf-8"
    )
    (root / "tier4" / "T001.json").write_text(
        json.dumps(_tier4_result("T001", "the disk is full")), encoding="utf-8"
    )

    with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
        tickets_by_id = {"T001": TICKET_A}
        scored = tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=True)

    assert len(captured_semantic_prompts) == 2
    forbidden = ["tier", "baseline", "agent loop", "orchestrator"]
    for prompt in captured_semantic_prompts:
        lowered = prompt.lower()
        for term in forbidden:
            assert term not in lowered
        for digit in "1234":
            # digits may appear incidentally (e.g. in ticket text numbers),
            # but never as a bare tier marker -- check the specific phrase.
            assert f"tier {digit}" not in lowered
            assert f"tier{digit}" not in lowered

    # Same candidate text -> identical prompt regardless of which tier
    # produced it.
    assert captured_semantic_prompts[0] == captured_semantic_prompts[1]


# ---------------------------------------------------------------------------
# Abstention for tier 4 uses the prose judge, not state.status
# ---------------------------------------------------------------------------


def test_tier4_abstention_uses_prose_not_status(tmp_path):
    root = tmp_path / "root"
    (root / "tier4").mkdir(parents=True)
    result = _tier4_result(
        "T900", "The root cause is definitely a misconfigured load balancer health check.", status="escalated"
    )
    (root / "tier4" / "T900.json").write_text(json.dumps(result), encoding="utf-8")

    def fake_abstention(ticket_text, candidate_answer):
        return False, "answer names a specific cause confidently", None

    with patch.object(llm_judge, "judge_abstention_once", side_effect=fake_abstention):
        tickets_by_id = {"T900": TICKET_AMBIG}
        scored = tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=True)

    cell = scored["abstention_cells"][4]
    assert cell["n_success_total"] == 0
    assert cell["n_trials_total"] == 1


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_caching_only_calls_judge_first_time(tmp_path):
    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_baseline_result(1, "T001", "the disk is full")), encoding="utf-8"
    )

    calls = {"n": 0}

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        calls["n"] += 1
        return True, "ok", None

    tickets_by_id = {"T001": TICKET_A}

    with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
        tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=False)
        assert calls["n"] == 1
        tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=False)
        assert calls["n"] == 1  # cache hit, no new call


# ---------------------------------------------------------------------------
# --live gate
# ---------------------------------------------------------------------------


def test_live_gate_exits_2_without_live(tmp_path, capsys):
    with patch.object(llm_judge, "judge_semantic_match_once") as mock_judge:
        code = tier_compare.main(["--root", str(tmp_path)])
        assert code == 2
        mock_judge.assert_not_called()


# ---------------------------------------------------------------------------
# Crashed results
# ---------------------------------------------------------------------------


def test_crashed_results_count_as_not_correct(tmp_path):
    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    crashed = _baseline_result(1, "T001", None)
    crashed["baseline"] = None
    (root / "tier1" / "T001.json").write_text(json.dumps(crashed), encoding="utf-8")

    tickets_by_id = {"T001": TICKET_A}
    scored = tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=True)

    assert scored["n_crashed"][1] == 1
    cell = scored["primary_cells"][1]["overall"]
    assert cell["n_success_total"] == 0
    assert cell["n_trials_total"] == 1


# ---------------------------------------------------------------------------
# tier4_only markers
# ---------------------------------------------------------------------------


def test_tier4_only_carries_na_marker(tmp_path):
    root = tmp_path / "root"
    (root / "tier4").mkdir(parents=True)
    (root / "tier4" / "T001.json").write_text(
        json.dumps(_tier4_result("T001", "the disk is full")), encoding="utf-8"
    )
    tickets_by_id = {"T001": TICKET_A}
    records = tier_compare.discover_results(root, tiers=None)
    tier4_only = tier_compare.build_tier4_only(records, tickets_by_id)
    for name in tier_compare.TIER4_ONLY_METRICS:
        assert tier4_only[name]["tiers_1_3"] == "n/a (no TaskState)"


def _tier4_result_with_calls(ticket_id, hypothesis, trajectory, status="resolved", repeat=0):
    result = _tier4_result(ticket_id, hypothesis, status=status, repeat=repeat)
    result["state"]["trajectory"] = trajectory
    return result


def test_tier4_only_parameter_validity_is_structured_not_bare_int(tmp_path):
    root = tmp_path / "root"
    (root / "tier4").mkdir(parents=True)
    trajectory = [
        {"tool_call": {"tool": "query_logs"}, "observation": {"status": "ok"}, "iteration": 0},
        {"tool_call": {"tool": "query_metrics"}, "observation": {"status": "invalid_args"}, "iteration": 0},
    ]
    (root / "tier4" / "T001.json").write_text(
        json.dumps(_tier4_result_with_calls("T001", "the disk is full", trajectory)), encoding="utf-8"
    )
    tickets_by_id = {"T001": TICKET_A}
    records = tier_compare.discover_results(root, tiers=None)
    tier4_only = tier_compare.build_tier4_only(records, tickets_by_id)
    entry = tier4_only["parameter_validity"]
    assert not isinstance(entry, int)
    assert entry["observed_failures"] == 1
    assert entry["total_calls"] == 2
    assert entry["measurable"] is False
    assert "note" in entry
    assert "not a percentage" in entry["note"] or "never a percentage" in entry["note"].lower() or "NEVER a percentage" in entry["note"]
    assert entry["tiers_1_3"] == "n/a (no TaskState)"


def test_tier4_only_parameter_validity_sums_across_multiple_results(tmp_path):
    root = tmp_path / "root"
    (root / "tier4").mkdir(parents=True)
    traj_a = [
        {"tool_call": {"tool": "query_logs"}, "observation": {"status": "ok"}, "iteration": 0},
        {"tool_call": {"tool": "query_metrics"}, "observation": {"status": "invalid_args"}, "iteration": 0},
    ]
    traj_b = [
        {"tool_call": {"tool": "query_logs"}, "observation": {"status": "ok"}, "iteration": 0},
        {"tool_call": {"tool": "query_logs"}, "observation": {"status": "ok"}, "iteration": 0},
        {"tool_call": {"tool": "query_metrics"}, "observation": {"status": "argument_error"}, "iteration": 0},
    ]
    traj_c = [
        {"tool_call": {"tool": "query_logs"}, "observation": {"status": "ok"}, "iteration": 0},
    ]
    (root / "tier4" / "T001.json").write_text(
        json.dumps(_tier4_result_with_calls("T001", "hyp a", traj_a)), encoding="utf-8"
    )
    (root / "tier4" / "T002.json").write_text(
        json.dumps(_tier4_result_with_calls("T002", "hyp b", traj_b)), encoding="utf-8"
    )
    (root / "tier4" / "T003.json").write_text(
        json.dumps(_tier4_result_with_calls("T003", "hyp c", traj_c)), encoding="utf-8"
    )
    tickets_by_id = {
        "T001": {**TICKET_A, "id": "T001"},
        "T002": {**TICKET_A, "id": "T002"},
        "T003": {**TICKET_A, "id": "T003"},
    }
    records = tier_compare.discover_results(root, tiers=None)
    tier4_only = tier_compare.build_tier4_only(records, tickets_by_id)
    entry = tier4_only["parameter_validity"]
    assert entry["observed_failures"] == 2
    assert entry["total_calls"] == 6


def test_tier4_only_parameter_validity_renders_as_not_a_rate():
    from eval import report as report_mod

    safety = {
        "status": "PASS",
        "unauthorized_write_block_rate": "n/a",
        "injection_block_rate": "n/a",
    }
    tc = {
        "summary": "",
        "primary": {},
        "comparisons": {"primary": {}},
        "abstention": {},
        "efficiency": {"rate_card": {"provider": "n/a"}, "tiers": {}},
        "tier4_only": {
            "parameter_validity": {
                "observed_failures": 0,
                "total_calls": 1761,
                "measurable": False,
                "note": "measurable=False by design -- never a percentage",
                "tiers_1_3": "n/a (no TaskState)",
            }
        },
    }
    report = {
        "safety": safety,
        "summary": "x",
        "provenance": {"n_tickets_scored": 0, "earliest_started_at": None, "latest_started_at": None},
        "rate_card": {"provider": "n/a", "used": False},
        "task_success": {
            "task_success_status_only": {"n_success": 0, "n_total": 0, "rate": None},
            "task_success_strict_lexical": {"n_success": 0, "n_total": 0, "rate": None},
            "hypothesis_semantic": {"status": "NOT COMPUTED", "reason": "n/a"},
            "gap_explanation": "n/a",
        },
        "category_breakdown": {
            cat: {
                "n": 0,
                "task_success_status_only": {"n_success": 0, "rate": None},
                "task_success_strict_lexical": {"n_success": 0, "rate": None},
            }
            for cat in report_mod.CATEGORIES
        },
        "known_issues": [],
        "stale_known_issues": [],
        "unexplained_failures": [],
        "crashed": {"n": 0, "tickets": []},
        "efficiency": {
            field: {"mean": None, "p50": None, "p95": None, "n": 0}
            for field in (
                "llm_call_count", "tool_call_count", "total_tokens_in",
                "total_tokens_out", "wall_clock_seconds", "estimated_cost_usd",
            )
        },
        "tool_use": {
            "tool_selection_accuracy": {"mean": None, "n": 0},
            "unnecessary_tool_calls_total": 0,
            "parameter_validity": {"observed_failures": 0, "total_calls": 0, "measurable": False, "note": ""},
        },
        "state_tracking": {
            "state_consistency": {"n_true": 0, "n": 0, "rate": None},
            "write_gate_appended_correctly": {"n_true": 0, "n": 0, "rate": None, "note": ""},
        },
        "rag": {
            "retrieval_recall_at_3_observed": {"n_true": 0, "n": 0, "rate": None},
            "citation_presence_rate": {"n_true": 0, "n": 0, "rate": None},
            "retrieval_recall_at_3_corpus": None,
        },
        "stale_gold_warnings": [],
        "missing_gold_tickets": [],
        "per_ticket": [],
        "tier_comparison": tc,
    }
    md = report_mod.render_markdown(report)
    assert "not a rate" in md
    assert "0 argument-validation failures in 1761 tool calls" in md
    # Must not present it as a bare number in a rate-like column, e.g. "tier4=0"
    assert "tier4=0" not in md


# ---------------------------------------------------------------------------
# All-inconclusive summary
# ---------------------------------------------------------------------------


def test_all_inconclusive_produces_plain_words_summary():
    cell4 = tier_compare.aggregate_cell([(5, 10)])
    other = tier_compare.aggregate_cell([(5, 10)])
    comparisons = {
        "primary": {1: {"overall": tier_compare.compare_to_tier4(cell4, other)}},
        "abstention": None,
    }
    assert tier_compare.all_inconclusive(comparisons) is True


def test_discover_results_no_repeat_subdir(tmp_path):
    root = tmp_path / "root"
    (root / "tier2").mkdir(parents=True)
    (root / "tier2" / "T001.json").write_text(
        json.dumps(_baseline_result(2, "T001", "answer", repeat=3)), encoding="utf-8"
    )
    records = tier_compare.discover_results(root, tiers=None)
    assert len(records) == 1
    assert records[0]["tier"] == 2
    assert records[0]["repeat"] == 3


# ---------------------------------------------------------------------------
# Regression: ordinary IT vocabulary in a real candidate answer must not
# raise. The old content scan (_FORBIDDEN_BLIND_TERMS / assert_blind) was
# removed because it treated words like "orchestrator", "baseline", and
# "tier 1" appearing in genuine model-generated prose as a blind-leak, when
# in fact the candidate answer's content can never reveal which tier
# produced it -- only the (tier-blind) prompt builders matter for blinding.
# ---------------------------------------------------------------------------


def test_ordinary_it_vocabulary_in_answer_does_not_raise(tmp_path):
    real_answers = [
        "In Kubernetes (or similar orchestrators), a canary rollout will not "
        "advance until the first replica passes its readiness probe.",
        "Latency spiked well above baseline traffic levels during the incident window.",
        "This was routed to tier 1 support before being escalated.",
    ]

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        return True, "ok", None

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    for i, answer in enumerate(real_answers):
        ticket_id = f"T00{i}"
        (root / "tier1" / f"{ticket_id}.json").write_text(
            json.dumps(_baseline_result(1, ticket_id, answer)), encoding="utf-8"
        )

    tickets_by_id = {
        f"T00{i}": {**TICKET_A, "id": f"T00{i}"} for i in range(len(real_answers))
    }

    with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
        # Must not raise.
        scored = tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=True)

    assert scored["n_crashed"].get(1, 0) == 0
    assert len(scored["per_answer"]) == len(real_answers)
    for entry in scored["per_answer"]:
        assert entry["verdict"] is True


# ---------------------------------------------------------------------------
# Structural blinding guard
# ---------------------------------------------------------------------------


def test_assert_prompt_tier_invariant_passes_for_current_builders():
    # Must not raise for the real llm_judge builders.
    tier_compare.assert_prompt_tier_invariant()


def test_assert_prompt_tier_invariant_fails_if_builder_gains_tier_param():
    def fake_build_prompt_3a(ticket_text, category, hypothesis, gold_root_cause, tier):
        return "prompt"

    with patch.object(llm_judge, "build_prompt_3a", fake_build_prompt_3a):
        with pytest.raises(AssertionError):
            tier_compare.assert_prompt_tier_invariant()


# ---------------------------------------------------------------------------
# Incremental cache persistence / robustness
# ---------------------------------------------------------------------------


def _make_many_tickets(root: Path, n: int):
    tier_dir = root / "tier1"
    tier_dir.mkdir(parents=True, exist_ok=True)
    tickets_by_id = {}
    for i in range(n):
        ticket_id = f"T{i:04d}"
        (tier_dir / f"{ticket_id}.json").write_text(
            json.dumps(_baseline_result(1, ticket_id, f"answer {i}")), encoding="utf-8"
        )
        tickets_by_id[ticket_id] = {
            "id": ticket_id,
            "category": "easy",
            "ticket_text": f"ticket text {i}",
            "gold_root_cause": "some_cause",
            "expected_behavior": "resolve_with_approval",
        }
    return tickets_by_id


def test_cache_flushed_periodically_for_long_run(tmp_path):
    root = tmp_path / "root"
    n = tier_compare.CACHE_FLUSH_EVERY * 2 + 5
    tickets_by_id = _make_many_tickets(root, n)

    save_calls = {"n": 0}
    real_save_cache = tier_compare.save_cache

    def fake_save_cache(path, cache):
        save_calls["n"] += 1
        real_save_cache(path, cache)

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        return True, "ok", None

    with patch.object(tier_compare, "save_cache", side_effect=fake_save_cache):
        with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
            tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=False)

    assert save_calls["n"] > 1


def test_cache_saved_once_for_short_run(tmp_path):
    root = tmp_path / "root"
    n = tier_compare.CACHE_FLUSH_EVERY - 5
    tickets_by_id = _make_many_tickets(root, n)

    save_calls = {"n": 0}
    real_save_cache = tier_compare.save_cache

    def fake_save_cache(path, cache):
        save_calls["n"] += 1
        real_save_cache(path, cache)

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        return True, "ok", None

    with patch.object(tier_compare, "save_cache", side_effect=fake_save_cache):
        with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
            tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=False)

    assert save_calls["n"] == 1


def test_save_cache_is_atomic_and_round_trips(tmp_path):
    cache_path = tmp_path / "judge_cache.json"
    cache = {"key1": {"verdict": True, "reasoning": "r", "error": None}}

    tier_compare.save_cache(cache_path, cache)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []
    assert cache_path.exists()
    loaded = json.loads(cache_path.read_text(encoding="utf-8"))
    assert loaded == cache


def test_load_cache_corrupt_file_starts_empty_and_warns(tmp_path, capsys):
    cache_path = tmp_path / "judge_cache.json"
    cache_path.write_text("{not valid json!!", encoding="utf-8")

    result = tier_compare.load_cache(cache_path)

    assert result == {}
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_corrupt_cache_does_not_abort_run(tmp_path, capsys):
    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_baseline_result(1, "T001", "the disk is full")), encoding="utf-8"
    )
    (root / "judge_cache.json").write_text("garbage not json", encoding="utf-8")

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        return True, "ok", None

    with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
        scored = tier_compare.score_all(root, {"T001": TICKET_A}, tiers=None, no_cache=False)

    assert scored["n_judge_calls_made"] == 1
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_progress_lines_go_to_stderr_not_stdout(tmp_path, capsys):
    root = tmp_path / "root"
    n = tier_compare.CACHE_FLUSH_EVERY + 3
    tickets_by_id = _make_many_tickets(root, n)

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        return True, "ok", None

    with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
        tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=False)

    captured = capsys.readouterr()
    assert "[judged" not in captured.out
    assert "[judged" in captured.err


def test_interrupted_run_keeps_earlier_verdicts(tmp_path):
    root = tmp_path / "root"
    n = tier_compare.CACHE_FLUSH_EVERY + 5
    tickets_by_id = _make_many_tickets(root, n)

    calls = {"n": 0}
    crash_at = tier_compare.CACHE_FLUSH_EVERY + 2

    def flaky_semantic(ticket_text, category, hypothesis, gold_root_cause):
        calls["n"] += 1
        if calls["n"] == crash_at:
            raise RuntimeError("simulated interrupt")
        return True, "ok", None

    with patch.object(llm_judge, "judge_semantic_match_once", side_effect=flaky_semantic):
        with pytest.raises(RuntimeError):
            tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=False)

    cache_path = root / "judge_cache.json"
    assert cache_path.exists()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(cache) >= tier_compare.CACHE_FLUSH_EVERY


# ---------------------------------------------------------------------------
# Ticket-set inference from result files (run.tickets_path / tickets_sha256)
# ---------------------------------------------------------------------------


def _write_temp_tickets(path: Path, tickets: list[dict]) -> str:
    path.write_text(json.dumps(tickets), encoding="utf-8")
    return tickets_content_sha256(path)


def _schema4_result(tier, ticket_id, answer, tickets_path, tickets_sha256, repeat=0):
    return {
        "schema_version": 4,
        "tier": tier,
        "repeat": repeat,
        "ticket_id": ticket_id,
        "run": {
            "wall_clock_seconds": 1.0,
            "tickets_path": tickets_path,
            "tickets_sha256": tickets_sha256,
        },
        "usage": {"llm_call_count": 1, "total_tokens_in": 10, "total_tokens_out": 10},
        "baseline": {"final_answer": answer} if answer is not None else None,
    }


def test_resolve_ticket_source_all_agree_scores_against_temp_file(tmp_path, capsys):
    temp_tickets_path = tmp_path / "temp_tickets.json"
    temp_tickets = [
        {
            "id": "T001",
            "category": "easy",
            "ticket_text": "Checkout API returning 500s under load.",
            "gold_root_cause": "totally_different_gold_cause_from_temp_file",
            "expected_behavior": "resolve_with_approval",
        }
    ]
    sha = _write_temp_tickets(temp_tickets_path, temp_tickets)

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(
            _schema4_result(1, "T001", "the answer", str(temp_tickets_path), sha)
        ),
        encoding="utf-8",
    )

    resolved = tier_compare.resolve_ticket_source(root)
    assert resolved == temp_tickets_path
    captured = capsys.readouterr()
    assert str(temp_tickets_path) in captured.err
    assert sha in captured.err

    # Prove scoring actually reflects the temp file's gold, not
    # data/tickets.json's -- this is the test that actually proves the fix.
    captured_gold = []

    def fake_semantic(ticket_text, category, hypothesis, gold_root_cause):
        captured_gold.append(gold_root_cause)
        return True, "ok", None

    tickets_by_id = tier_compare.load_tickets(resolved)
    with patch.object(llm_judge, "judge_semantic_match_once", side_effect=fake_semantic):
        tier_compare.score_all(root, tickets_by_id, tiers=None, no_cache=True)

    assert captured_gold == ["totally_different_gold_cause_from_temp_file"]


def test_resolve_ticket_source_mixed_sha256_aborts(tmp_path):
    tickets_a = tmp_path / "a.json"
    tickets_b = tmp_path / "b.json"
    sha_a = _write_temp_tickets(tickets_a, [{"id": "T001"}])
    sha_b = _write_temp_tickets(tickets_b, [{"id": "T002"}])

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_schema4_result(1, "T001", "a1", str(tickets_a), sha_a)), encoding="utf-8"
    )
    (root / "tier1" / "T002.json").write_text(
        json.dumps(_schema4_result(1, "T002", "b1", str(tickets_b), sha_b)), encoding="utf-8"
    )

    with pytest.raises(tier_compare.TicketSourceError) as excinfo:
        tier_compare.resolve_ticket_source(root)

    msg = str(excinfo.value)
    assert sha_a in msg
    assert sha_b in msg
    assert "1 file" in msg


def test_resolve_ticket_source_mixed_schema_aborts(tmp_path):
    tickets_a = tmp_path / "a.json"
    sha_a = _write_temp_tickets(tickets_a, [{"id": "T001"}])

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_schema4_result(1, "T001", "a1", str(tickets_a), sha_a)), encoding="utf-8"
    )
    # Pre-4 result, no run.tickets_path/tickets_sha256.
    (root / "tier1" / "T002.json").write_text(
        json.dumps(_baseline_result(1, "T002", "old answer")), encoding="utf-8"
    )

    with pytest.raises(tier_compare.TicketSourceError):
        tier_compare.resolve_ticket_source(root)


def test_resolve_ticket_source_all_pre4_falls_back_to_default(tmp_path, capsys):
    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_baseline_result(1, "T001", "old answer")), encoding="utf-8"
    )

    resolved = tier_compare.resolve_ticket_source(root)
    assert resolved == tier_compare.TICKETS_PATH
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "schema_version 4" in captured.err


def test_resolve_ticket_source_recorded_path_missing_aborts(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    sha = hashlib.sha256(b"[]").hexdigest()

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_schema4_result(1, "T001", "a1", str(missing_path), sha)), encoding="utf-8"
    )

    with pytest.raises(tier_compare.TicketSourceError) as excinfo:
        tier_compare.resolve_ticket_source(root)
    assert "does not exist" in str(excinfo.value) or "no longer exist" in str(excinfo.value)


def test_resolve_ticket_source_override_wins_and_warns(tmp_path, capsys):
    override_path = tmp_path / "override.json"
    override_path.write_text("[]", encoding="utf-8")

    root = tmp_path / "root"  # does not even need to exist

    resolved = tier_compare.resolve_ticket_source(root, tickets_path_override=str(override_path))
    assert resolved == override_path
    captured = capsys.readouterr()
    assert "override" in captured.err.lower()


def test_resolve_ticket_source_override_missing_file_aborts(tmp_path):
    override_path = tmp_path / "nope.json"
    root = tmp_path / "root"

    with pytest.raises(tier_compare.TicketSourceError):
        tier_compare.resolve_ticket_source(root, tickets_path_override=str(override_path))


def test_resolve_ticket_source_line_ending_variants_not_treated_as_mixed(tmp_path, capsys):
    # Same logical ticket content, written with CRLF and with LF, at two
    # different paths. Hashed with the shared content-based helper, these
    # must agree -- the regression this guards against is raw-byte hashing
    # treating a CRLF checkout and an LF checkout of the same tickets.json
    # as a "mixed ticket set" and aborting a valid comparison.
    tickets = [{"id": "T001", "category": "easy"}]
    text = json.dumps(tickets)

    crlf_path = tmp_path / "crlf_tickets.json"
    lf_path = tmp_path / "lf_tickets.json"
    crlf_path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    lf_path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))

    sha_crlf = tickets_content_sha256(crlf_path)
    sha_lf = tickets_content_sha256(lf_path)
    assert sha_crlf == sha_lf

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_schema4_result(1, "T001", "crlf answer", str(crlf_path), sha_crlf)),
        encoding="utf-8",
    )
    (root / "tier1" / "T002.json").write_text(
        json.dumps(_schema4_result(1, "T002", "lf answer", str(lf_path), sha_lf)),
        encoding="utf-8",
    )

    # Both result files agree on tickets_sha256 but disagree on
    # tickets_path -- resolve_ticket_source should still refuse (a
    # tickets_path mismatch is a real disagreement), but importantly it
    # must NOT abort at the sha256 stage, proving the two line-ending
    # variants were not treated as a "mixed ticket set" on hash grounds.
    with pytest.raises(tier_compare.TicketSourceError) as excinfo:
        tier_compare.resolve_ticket_source(root)
    assert "disagree on tickets_path" in str(excinfo.value)


def test_main_without_live_runs_ticket_inference_first(tmp_path, capsys):
    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_baseline_result(1, "T001", "old answer")), encoding="utf-8"
    )
    code = tier_compare.main(["--root", str(root)])
    assert code == 2
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_main_aborts_nonzero_on_mixed_ticket_source(tmp_path, capsys):
    tickets_a = tmp_path / "a.json"
    tickets_b = tmp_path / "b.json"
    sha_a = _write_temp_tickets(tickets_a, [{"id": "T001"}])
    sha_b = _write_temp_tickets(tickets_b, [{"id": "T002"}])

    root = tmp_path / "root"
    (root / "tier1").mkdir(parents=True)
    (root / "tier1" / "T001.json").write_text(
        json.dumps(_schema4_result(1, "T001", "a1", str(tickets_a), sha_a)), encoding="utf-8"
    )
    (root / "tier1" / "T002.json").write_text(
        json.dumps(_schema4_result(1, "T002", "b1", str(tickets_b), sha_b)), encoding="utf-8"
    )

    code = tier_compare.main(["--root", str(root), "--live"])
    assert code != 0
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
