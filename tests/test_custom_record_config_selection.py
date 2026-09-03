#!/usr/bin/env python3
"""Tests for --review-config selection + recovery path guard in custom_record_final.py.

Pure logic tests only (config resolution + recovery guard) with temp YAMLs.
No robot, no teleop, no camera, no CAN, no recording.

  TEST 1  default (no --review-config) -> historical record_review_config.yaml,
          guard OFF, never REFUSEs even though the default roots still contain
          the legacy 184850 marker.
  TEST 2  explicit --review-config record_review_config_recovery_v1.yaml loads
          the recovery YAML (raw/reviewed roots point at piper_openpi_v21_recovery_v1).
  TEST 3  recovery guard ON + correct roots -> PASS.
  TEST 4  recovery guard ON + legacy/incorrect roots -> REFUSE.
  TEST 5  a custom NON-recovery YAML is not blocked by recovery name hard-coding.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import episode_tools
# Guard logic lives in record_review_guard.py (import-light, no lerobot chain),
# so these tests run in the deploy env without importing custom_record_final.
import record_review_guard as crf

LEGACY_RAW = "<DATASET_ROOT>/piper_demo_review"
LEGACY_REVIEWED = "<DATASET_ROOT>/piper_demo_review_reviewed"
RECOVERY_RAW = "<DATASET_ROOT>/piper_openpi_v21_recovery_v1"
RECOVERY_REVIEWED = "<DATASET_ROOT>/piper_openpi_v21_recovery_v1_reviewed"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "recording"
RECOVERY_YAML = SCRIPT_DIR / "record_review_config_recovery_v1.yaml"


def _write_yaml(dirpath: str, name: str, raw_root: str, reviewed_root: str) -> pathlib.Path:
    import yaml

    p = pathlib.Path(dirpath) / name
    p.write_text(
        yaml.safe_dump(
            {
                "fps": 30,
                "recording": {
                    "auto_approve_after_stop": False,
                    "auto_trim_on_accept": True,
                    "raw_root": raw_root,
                    "reviewed_root": reviewed_root,
                    "collection_stage": "FIRST_5",
                },
            },
            allow_unicode=True,
        )
    )
    return p


def test_1_default_no_review_config_guard_off():
    cfg = episode_tools.load_config()  # no path -> script-dir record_review_config.yaml
    raw = cfg["recording"]["raw_root"]
    rev = cfg["recording"]["reviewed_root"]
    assert LEGACY_RAW in raw, f"default raw_root must still be legacy, got {raw!r}"
    # no --review-config -> guard OFF, never REFUSE regardless of paths
    guard_on, ok, reason = crf.recovery_guard(None, raw, rev)
    assert guard_on is False and ok is True, (guard_on, ok, reason)
    print("TEST 1 PASS: DEFAULT_BEHAVIOR_UNCHANGED, guard OFF for default config")


def test_2_recovery_yaml_loads():
    assert RECOVERY_YAML.exists(), f"missing {RECOVERY_YAML}"
    cfg = episode_tools.load_config(RECOVERY_YAML)
    assert cfg["recording"]["raw_root"] == RECOVERY_RAW, cfg["recording"]["raw_root"]
    assert cfg["recording"]["reviewed_root"] == RECOVERY_REVIEWED, cfg["recording"]["reviewed_root"]
    print("TEST 2 PASS: recovery YAML loaded; roots -> piper_openpi_v21_recovery_v1")


def test_3_recovery_guard_ok_with_correct_roots():
    guard_on, ok, reason = crf.recovery_guard(RECOVERY_YAML, RECOVERY_RAW, RECOVERY_REVIEWED)
    assert guard_on is True, "explicit recovery-named config must turn guard ON"
    assert ok is True, reason
    print("TEST 3 PASS: guard ON + correct roots -> PASS")


def test_4_recovery_guard_refuses_legacy_roots():
    bad = [(LEGACY_RAW, LEGACY_REVIEWED),
           (RECOVERY_RAW, LEGACY_REVIEWED),
           (LEGACY_RAW, RECOVERY_REVIEWED)]
    for raw, rev in bad:
        guard_on, ok, reason = crf.recovery_guard(RECOVERY_YAML, raw, rev)
        assert guard_on is True and ok is False, (raw, rev, guard_on, ok, reason)
    print("TEST 4 PASS: guard ON + legacy/incorrect roots -> REFUSE")


def test_5_custom_non_recovery_yaml_not_blocked():
    with tempfile.TemporaryDirectory() as td:
        p = _write_yaml(td, "custom_pipeline_check.yaml", LEGACY_RAW, LEGACY_REVIEWED)
        cfg = episode_tools.load_config(p)
        assert cfg["recording"]["raw_root"] == LEGACY_RAW
        guard_on, ok, reason = crf.recovery_guard(
            p, cfg["recording"]["raw_root"], cfg["recording"]["reviewed_root"])
        assert guard_on is False and ok is True, (guard_on, ok, reason)
    print("TEST 5 PASS: custom non-recovery YAML loads, guard OFF, not blocked")


if __name__ == "__main__":
    tests = [test_1_default_no_review_config_guard_off,
             test_2_recovery_yaml_loads,
             test_3_recovery_guard_ok_with_correct_roots,
             test_4_recovery_guard_refuses_legacy_roots,
             test_5_custom_non_recovery_yaml_not_blocked]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[{fn.__name__}] PASS")
        except Exception as e:
            print(f"[{fn.__name__}] FAIL: {type(e).__name__}: {e}")
            failed += 1
    print(f"\nTESTS_PASSED = {len(tests) - failed}/{len(tests)}")
    sys.exit(1 if failed else 0)
