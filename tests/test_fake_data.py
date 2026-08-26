import json

import pytest

import tools.fake_data as fake_data
from tools.fake_data import (
    METRIC_PROFILES,
    ROOT_CAUSE_SIGNATURES,
    TICKETS_PATH,
    TICKETS_PATH_ENV,
    get_ticket,
    load_tickets,
    resolve_tickets_path,
    signature_for_ticket,
)

REQUIRED_KEYS = {"log_patterns", "log_level", "metric_name", "metric_behavior"}


def _all_tickets():
    return json.loads(TICKETS_PATH.read_text(encoding="utf-8"))


def test_every_gold_root_cause_has_a_signature():
    tickets = _all_tickets()
    gold_causes = {t["gold_root_cause"] for t in tickets}
    # includes the literal None for the two ambiguous tickets
    assert None in gold_causes
    for cause in gold_causes:
        assert cause in ROOT_CAUSE_SIGNATURES, f"missing signature for {cause!r}"


def test_signature_shape():
    for cause, sig in ROOT_CAUSE_SIGNATURES.items():
        assert set(sig.keys()) == REQUIRED_KEYS, f"{cause!r} has wrong keys: {sig.keys()}"
        assert isinstance(sig["log_patterns"], list)
        assert len(sig["log_patterns"]) > 0
        for pattern in sig["log_patterns"]:
            assert isinstance(pattern, str)
            assert pattern.strip()
        assert isinstance(sig["log_level"], str)
        assert isinstance(sig["metric_name"], str)
        assert isinstance(sig["metric_behavior"], str)
        assert sig["metric_behavior"].strip()


def test_signature_metric_name_is_a_known_metric():
    for cause, sig in ROOT_CAUSE_SIGNATURES.items():
        assert sig["metric_name"] in METRIC_PROFILES, (
            f"{cause!r} references unknown metric {sig['metric_name']!r}"
        )


def test_log_patterns_render_without_keyerror():
    for cause, sig in ROOT_CAUSE_SIGNATURES.items():
        for pattern in sig["log_patterns"]:
            rendered = pattern.format(timestamp="2025-01-01T00:00:00", service="checkout")
            assert "{timestamp}" not in rendered
            assert "{service}" not in rendered


def test_signature_for_ticket_real_cause():
    sig = signature_for_ticket("T001")
    assert sig is ROOT_CAUSE_SIGNATURES["db_connection_pool_exhaustion"]


def test_signature_for_ticket_ambiguous():
    sig = signature_for_ticket("T014")
    assert sig is ROOT_CAUSE_SIGNATURES[None]


def test_signature_for_ticket_unknown_id():
    assert signature_for_ticket("T999") is None


# --- Round 3: METRIC_PROFILES direction/shape/integer -----------------------

VALID_DIRECTIONS = {"high_bad", "low_bad"}
VALID_SHAPES = {"sustained", "climbing", "step_up", "step_down"}


def test_every_metric_profile_has_direction_shape_integer():
    for metric, profile in METRIC_PROFILES.items():
        assert "direction" in profile, f"{metric!r} missing 'direction'"
        assert profile["direction"] in VALID_DIRECTIONS, f"{metric!r} bad direction"
        assert "shape" in profile, f"{metric!r} missing 'shape'"
        assert profile["shape"] in VALID_SHAPES, f"{metric!r} bad shape"
        assert "integer" in profile, f"{metric!r} missing 'integer'"
        assert isinstance(profile["integer"], bool)


def test_only_readiness_probe_is_low_bad():
    low_bad = {m for m, p in METRIC_PROFILES.items() if p["direction"] == "low_bad"}
    assert low_bad == {"readiness_probe_success_rate"}


def test_shape_matches_signature_metric_behavior_intent():
    expected_shapes = {
        "db_pool_active_connections": "sustained",
        "cache_used_memory_percent": "sustained",
        "ingress_listen_queue_depth": "sustained",
        "disk_usage_percent": "climbing",
        "auth_token_verification_failure_rate": "step_up",
        "readiness_probe_success_rate": "step_down",
        "request_latency_p95": "sustained",
    }
    for metric, shape in expected_shapes.items():
        assert METRIC_PROFILES[metric]["shape"] == shape


def test_ingress_listen_queue_depth_is_a_connection_count():
    profile = METRIC_PROFILES["ingress_listen_queue_depth"]
    assert profile["unit"] == "connections"
    assert profile["capacity_max"] == 512.0
    assert profile["normal_range"] == (10.0, 60.0)
    assert profile["anomaly_range"] == (435.0, 512.0)
    assert profile["integer"] is True


def test_integer_metrics_flagged():
    integer_metrics = {m for m, p in METRIC_PROFILES.items() if p["integer"]}
    assert integer_metrics == {
        "db_pool_active_connections",
        "ingress_listen_queue_depth",
        "request_count",
    }


# --- Switchable ticket-set path -----------------------------------------


@pytest.fixture(autouse=True)
def _clear_ticket_env_and_cache(monkeypatch):
    monkeypatch.delenv(TICKETS_PATH_ENV, raising=False)
    fake_data._load_tickets_cached.cache_clear()
    yield
    fake_data._load_tickets_cached.cache_clear()


def test_resolve_tickets_path_default_when_env_unset():
    assert resolve_tickets_path() == TICKETS_PATH
    assert load_tickets() == _all_tickets()


def test_env_var_switches_ticket_set(tmp_path, monkeypatch):
    alt_tickets = [{"id": "TX01", "gold_root_cause": "db_connection_pool_exhaustion"}]
    alt_path = tmp_path / "alt_tickets.json"
    alt_path.write_text(json.dumps(alt_tickets), encoding="utf-8")

    monkeypatch.setenv(TICKETS_PATH_ENV, str(alt_path))

    assert resolve_tickets_path() == alt_path
    assert load_tickets() == alt_tickets
    assert get_ticket("TX01") == alt_tickets[0]
    assert get_ticket("T001") is None


def test_env_var_nonexistent_path_raises(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setenv(TICKETS_PATH_ENV, str(missing))

    with pytest.raises(FileNotFoundError) as exc_info:
        load_tickets()

    message = str(exc_info.value)
    assert TICKETS_PATH_ENV in message
    assert str(missing) in message

    # Must not silently fall back to the default ticket set.
    default_ids = {t["id"] for t in _all_tickets()}
    try:
        result_ids = {t["id"] for t in load_tickets()}
    except FileNotFoundError:
        result_ids = None
    assert result_ids != default_ids


def test_env_var_relative_path_resolves_against_repo_root(tmp_path, monkeypatch):
    alt_tickets = [{"id": "TX02", "gold_root_cause": None}]
    rel_dir = fake_data.REPO_ROOT / "_tmp_test_ticket_set_dir"
    rel_dir.mkdir(exist_ok=True)
    alt_path = rel_dir / "alt.json"
    try:
        alt_path.write_text(json.dumps(alt_tickets), encoding="utf-8")

        monkeypatch.setenv(TICKETS_PATH_ENV, "_tmp_test_ticket_set_dir/alt.json")
        monkeypatch.chdir(tmp_path)

        resolved = resolve_tickets_path()
        assert resolved == alt_path
        assert load_tickets() == alt_tickets
    finally:
        alt_path.unlink(missing_ok=True)
        rel_dir.rmdir()


def test_env_var_empty_string_treated_as_unset(monkeypatch):
    monkeypatch.setenv(TICKETS_PATH_ENV, "")
    assert resolve_tickets_path() == TICKETS_PATH
    assert load_tickets() == _all_tickets()
