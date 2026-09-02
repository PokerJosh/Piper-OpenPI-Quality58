"""Pure-function regression tests for the offline Quality58 H20 evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from eval_joint_pi05_rtc_h20_quality58_final_offline import (
    OFFLINE_GUARDS,
    baseline_comparison,
    summarize_errors,
    validate_action_shapes,
)


def test_offline_guards_disable_all_hardware_paths():
    assert OFFLINE_GUARDS == {
        "robot": "NO",
        "can": "NO",
        "camera": "NO",
        "motion": "NO",
    }


def test_validate_action_shapes_requires_physical_7d_and_model_32d():
    physical = np.zeros((20, 7), dtype=np.float32)
    assert validate_action_shapes(physical, model_action_dim=32) == (20, 7, 32)
    with pytest.raises(ValueError, match="physical 7D"):
        validate_action_shapes(np.zeros((20, 6), dtype=np.float32), model_action_dim=32)
    with pytest.raises(ValueError, match="model action dim 32"):
        validate_action_shapes(physical, model_action_dim=7)


def test_summary_and_missing_h20_baseline_are_explicitly_incomparable():
    summary = summarize_errors(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert summary["mae"] == pytest.approx(2.0)
    assert summary["p95"] == pytest.approx(2.9)
    comparison = baseline_comparison(final_metrics={"mae": 1.0}, baseline_metrics=None)
    assert comparison["comparable"] is False
    assert "H10/H20 incomparable" in comparison["status"]


def test_select_adjacent_pairs_uses_valid_consecutive_frames_and_covers_recovery():
    from eval_joint_pi05_rtc_h20_quality58_final_offline import select_adjacent_pairs

    window_index = []
    sources = {}
    for ep, source, count in (("nominal-0", "nominal", 120), ("recovery-0", "recovery", 20), ("recovery-1", "recovery", 20)):
        sources[ep] = source
        window_index.extend((ep, frame) for frame in range(count))
    pairs = select_adjacent_pairs(window_index, sources, min_pairs=100)
    assert len(pairs) >= 100
    assert {a[0] for a, _ in pairs if sources[a[0]] == "recovery"} == {"recovery-0", "recovery-1"}
    assert all(a[0] == b[0] and b[1] == a[1] + 1 for a, b in pairs)


def test_safety_diagnostics_report_units_and_expert_final_examples():
    from eval_joint_pi05_rtc_h20_quality58_final_offline import safety_diagnostics

    pred = np.zeros((2, 20, 7), dtype=np.float32)
    expert = np.zeros_like(pred)
    pred[0, 3, 2] = 1.0  # J3 is above its upper bound of 0 degrees.
    pred[1, 4, 6] = 0.08  # gripper is above its upper bound of 0.07 m.
    result = safety_diagnostics(pred, expert, [("ep-a", 10), ("ep-b", 20)])
    assert "direct_joint_core.py" in result["limit_source"]
    assert result["limit_units"] == {"joints": "degrees", "gripper": "meters"}
    assert result["final_j3_violation_count"] == 1
    assert result["expert_j3_violation_count"] == 0
    assert result["final_gripper_violation_count"] == 1
    assert result["expert_gripper_violation_count"] == 0
    assert result["j3_violation_examples"][0]["episode"] == "ep-a"
    assert result["j3_violation_examples"][0]["frame"] == 13
