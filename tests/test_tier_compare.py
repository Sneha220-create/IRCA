import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eval import llm_judge, tier_compare


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
