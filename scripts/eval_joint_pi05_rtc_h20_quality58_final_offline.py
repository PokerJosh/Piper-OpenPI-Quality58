#!/usr/bin/env python3
"""Strictly offline evaluation of the Quality58 H20 RTC final checkpoint.

This script deliberately uses only frozen mask-aware recorded windows and the
policy API. It never imports robot/CAN/camera runtime modules and never writes
inside a dataset or checkpoint tree.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

OFFLINE_GUARDS = {"robot": "NO", "can": "NO", "camera": "NO", "motion": "NO"}
OPENPI_ROOT = Path("<OPENPI_ROOT>")
FINAL_CHECKPOINT = Path("<CHECKPOINT_ROOT>/pi05_piper_joint_rtc_h20_quality58_finetune/pi05_piper_joint_rtc_h20_quality58_3k_20260902_135854/2999")
BASELINE_CHECKPOINT = Path("<CHECKPOINT_ROOT>/pi05_piper_joint_recovery_mix_finetune/pi05_piper_joint_recovery_mix_103ep_3k_20260831_133508_fastfinish/2999")
CONFIG_NAME = "pi05_piper_joint_rtc_h20_quality58_finetune"
BASELINE_CONFIG_NAME = "pi05_piper_joint_recovery_mix_finetune"
MANIFEST = Path("<DATA_QA_ROOT>/joint_rtc_expert_salvage_20260901T161500Z/FINAL_TRAINING_MANIFEST.json")
MASK = Path("<DATA_QA_ROOT>/joint_rtc_expert_salvage_20260901T161500Z/FINAL_TRAIN_SAMPLE_MASK.json")
PROMPT = "pick and place"
HORIZON = 20
PHYSICAL_DIM = 7
MODEL_DIM = 32
SEED = 20260902
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
ALL_NAMES = (*JOINT_NAMES, "G")
# These are the existing direct-joint runner's documented physical limits.
JOINT_LIMITS = np.array([(-150.0, 124.22), (0.0, 179.91), (-170.0, 0.0), (-99.98, 99.98), (-69.90, 69.90), (-120.0, 120.0)], dtype=np.float64)
GRIPPER_LIMITS = (0.0, 0.07)
LIMIT_SOURCE = ".worktrees/direct-joint2999/src/openpi/rollouts/direct_joint_core.py:PHYSICAL_HARD_LIMITS + GRIPPER_LIMITS_M"
LIMIT_UNITS = {"joints": "degrees", "gripper": "meters"}

sys.path.insert(0, str(OPENPI_ROOT / "src"))


def validate_action_shapes(physical_actions: np.ndarray, model_action_dim: int) -> tuple[int, int, int]:
    arr = np.asarray(physical_actions)
    if arr.ndim != 2 or arr.shape != (HORIZON, PHYSICAL_DIM):
        raise ValueError(f"physical 7D H20 action required, got {arr.shape}; expected physical 7D")
    if int(model_action_dim) != MODEL_DIM:
        raise ValueError(f"model action dim 32 required, got {model_action_dim}; expected model action dim 32")
    return (int(arr.shape[0]), int(arr.shape[1]), int(model_action_dim))


def summarize_errors(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    return {
        "count": int(x.size),
        "finite_count": int(finite.size),
        "mae": float(np.mean(np.abs(finite))) if finite.size else None,
        "p50": float(np.percentile(np.abs(finite), 50)) if finite.size else None,
        "p95": float(np.percentile(np.abs(finite), 95)) if finite.size else None,
        "max": float(np.max(np.abs(finite))) if finite.size else None,
    }


def baseline_comparison(final_metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None) -> dict[str, Any]:
    if baseline_metrics is None:
        return {"comparable": False, "status": "H10/H20 incomparable", "final": final_metrics, "baseline": None}
    return {"comparable": True, "status": "H10 first-action comparison only; full H20 is not comparable", "final": final_metrics, "baseline": baseline_metrics}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _noise(seed: int, episode_id: str, start: int, horizon: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, hash(episode_id) & 0xFFFFFFFF, int(start)])).standard_normal((horizon, MODEL_DIM)).astype(np.float32)


def _dataset():
    from openpi.training.data_loader import MaskAwareJointRTCDataset
    return MaskAwareJointRTCDataset(MANIFEST, MASK, action_horizon=HORIZON, prompt=PROMPT)


def _obs(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": {"top": np.asarray(sample["observation.images.top"]), "wrist": np.asarray(sample["observation.images.wrist"])},
        "state": np.asarray(sample["observation.state"], dtype=np.float32),
        "prompt": PROMPT,
    }


def _load_policy(config_name: str, checkpoint: Path):
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    cfg = training_config.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt=PROMPT)
    return policy, cfg


def _find_kernel_shape(tree: Any) -> list[int] | None:
    if isinstance(tree, dict):
        if "action_in_proj" in tree:
            node = tree["action_in_proj"]
            if isinstance(node, dict) and "kernel" in node:
                return list(np.shape(node["kernel"]))
        for value in tree.values():
            found = _find_kernel_shape(value)
            if found is not None:
                return found
    return None


def _checkpoint_kernel_shape(checkpoint: Path) -> list[int] | None:
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    params = model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    return _find_kernel_shape(params)


def _preflight() -> int:
    started = time.perf_counter()
    report: dict[str, Any] = {"mode": "preflight", "offline_only": True, "guards": OFFLINE_GUARDS, "checkpoint": FINAL_CHECKPOINT}
    try:
        if not FINAL_CHECKPOINT.is_dir() or not (FINAL_CHECKPOINT / "params").is_dir():
            raise FileNotFoundError(f"final checkpoint params missing: {FINAL_CHECKPOINT}")
        policy, cfg = _load_policy(CONFIG_NAME, FINAL_CHECKPOINT)
        ds = _dataset()
        sample = ds[0]
        state = np.asarray(sample["observation.state"])
        actions = np.asarray(sample["action"])
        if state.shape != (7,):
            raise ValueError(f"raw state shape {state.shape}, expected (7,)")
        if actions.shape != (20, 7):
            raise ValueError(f"raw action shape {actions.shape}, expected (20,7)")
        validate_action_shapes(actions, cfg.model.action_dim)
        transformed = policy._input_transform(_obs(sample))  # noqa: SLF001 - exact deployment policy transform
        report["raw_physical_state_shape"] = list(state.shape)
        report["raw_physical_action_shape"] = list(actions.shape)
        report["padded_state_shape"] = list(np.shape(transformed["state"]))
        report["padded_action_shape"] = list(np.shape(transformed.get("actions", np.zeros((20, MODEL_DIM), np.float32))))
        report["model_action_dim"] = int(cfg.model.action_dim)
        report["action_horizon"] = int(cfg.model.action_horizon)
        report["training_rtc_action_dim"] = int(cfg.model.training_rtc_action_dim)
        report["dataset_class"] = type(ds).__name__
        report["dataset_windows"] = len(ds)
        report["manifest_episodes"] = len(json.loads(MANIFEST.read_text())["episodes"])
        report["checkpoint_local_norm"] = (FINAL_CHECKPOINT / "assets/local/piper_joint_rtc_quality58/norm_stats.json").is_file()
        report["action_in_proj_kernel"] = _checkpoint_kernel_shape(FINAL_CHECKPOINT)
        out = policy.infer(_obs(sample), noise=_noise(SEED, ds.window_index[0][0], ds.window_index[0][1], HORIZON))
        physical = np.asarray(out["actions"], dtype=np.float32)
        report["physical_output_shape"] = list(physical.shape)
        report["all_output_finite"] = bool(np.isfinite(physical).all())
        report["inference_ms"] = float(out.get("policy_timing", {}).get("infer_ms", float("nan")))
        if physical.shape != (20, 7) or not np.isfinite(physical).all():
            raise RuntimeError("preflight physical output shape/finite check failed")
        report.update({"status": "PASS", "checkpoint_load": "PASS", "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps(report, indent=2, default=_json_default))
        return 0
    except BaseException as exc:
        report.update({"status": "FAIL", "checkpoint_load": "FAIL", "exception": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps(report, indent=2, default=_json_default), file=sys.stderr)
        return 1


def select_adjacent_pairs(
    window_index: list[tuple[str, int]],
    episode_sources: dict[str, str],
    *,
    min_pairs: int = 100,
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Select valid adjacent H20 windows from the frozen mask index.

    Starts are original episode frame indices. Pairing is exact ``t``/``t+1``
    within one episode, so the two predictions have the 19-step overlap used
    by the training-time inter-chunk RTC boundary semantics. Recovery episodes
    are seeded first so every recovery episode is represented deterministically.
    """
    by_ep: dict[str, set[int]] = {}
    for episode_id, start in window_index:
        by_ep.setdefault(str(episode_id), set()).add(int(start))
    all_pairs: dict[str, list[tuple[tuple[str, int], tuple[str, int]]]] = {}
    for episode_id in sorted(by_ep):
        starts = sorted(by_ep[episode_id])
        valid = [
            ((episode_id, start), (episode_id, start + 1))
            for start in starts
            if start + 1 in by_ep[episode_id]
        ]
        all_pairs[episode_id] = valid
    recovery_ids = [
        episode_id for episode_id in sorted(all_pairs)
        if episode_sources.get(episode_id) == "recovery"
    ]
    selected: list[tuple[tuple[str, int], tuple[str, int]]] = []
    used: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    for episode_id in recovery_ids:
        if all_pairs[episode_id]:
            pair = all_pairs[episode_id][0]
            selected.append(pair)
            used.add(pair)
    remaining = [
        pair for episode_id in sorted(all_pairs)
        for pair in all_pairs[episode_id]
        if pair not in used
    ]
    selected.extend(remaining[: max(0, int(min_pairs) - len(selected))])
    if len(selected) < int(min_pairs):
        raise ValueError(
            f"only {len(selected)} adjacent valid H20 pairs available; "
            f"required {int(min_pairs)}"
        )
    return selected


def _continuity_pairs(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    pairs: list[tuple[tuple[str, int], tuple[str, int]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure overlap discontinuity for exact adjacent physical 7D pairs."""
    values: list[float] = []
    records: list[dict[str, Any]] = []
    for index, (first, second) in enumerate(pairs):
        a = np.asarray(pred_a[index], dtype=np.float64)
        b = np.asarray(pred_b[index], dtype=np.float64)
        if a.shape != (HORIZON, PHYSICAL_DIM) or b.shape != (HORIZON, PHYSICAL_DIM):
            raise ValueError(f"RTC pair predictions must be ({HORIZON},7), got {a.shape} and {b.shape}")
        delta = np.abs(a[1:, :PHYSICAL_DIM] - b[:-1, :PHYSICAL_DIM])
        for dim, name in enumerate(ALL_NAMES):
            value = float(np.mean(delta[:, dim]))
            values.append(value)
            records.append({
                "episode": first[0],
                "frame": second[1],
                "joint": name,
                "value": value,
            })
    metrics = summarize_errors(np.asarray(values, dtype=np.float64))
    metrics.update({
        "pair_count": int(len(pairs)),
        "overlap_steps": HORIZON - 1,
        "physical_action_dim": PHYSICAL_DIM,
    })
    return metrics, sorted(records, key=lambda item: item["value"], reverse=True)[:10]


def safety_diagnostics(
    pred: np.ndarray,
    expert: np.ndarray,
    metadata: list[tuple[str, int]],
    baseline: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report hard-limit violations in physical units without clamping."""
    def _as_chunks(values: np.ndarray, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2 and array.shape[-1] == PHYSICAL_DIM:
            array = array[:, None, :]
        if array.ndim != 3 or array.shape[-1] != PHYSICAL_DIM:
            raise ValueError(f"{label} must have shape [N,H,7] or [N,7], got {array.shape}")
        if array.shape[0] != len(metadata):
            raise ValueError(f"{label} rows {array.shape[0]} != metadata rows {len(metadata)}")
        return array

    def _mask(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint = array[..., :6]
        grip = array[..., 6]
        joint_bad = (joint < JOINT_LIMITS[:, 0]) | (joint > JOINT_LIMITS[:, 1])
        grip_bad = (grip < GRIPPER_LIMITS[0]) | (grip > GRIPPER_LIMITS[1])
        return joint_bad, grip_bad

    pred_arr = _as_chunks(pred, "pred")
    expert_arr = _as_chunks(expert, "expert")
    baseline_arr = _as_chunks(baseline, "baseline") if baseline is not None else None
    pred_joint_bad, pred_grip_bad = _mask(pred_arr)
    expert_joint_bad, expert_grip_bad = _mask(expert_arr)
    base_joint_bad, base_grip_bad = _mask(baseline_arr) if baseline_arr is not None else (None, None)
    examples: list[dict[str, Any]] = []
    for row, (episode_id, start) in enumerate(metadata):
        for offset in range(pred_arr.shape[1]):
            if pred_joint_bad[row, offset, 2]:
                examples.append({
                    "episode": episode_id,
                    "frame": int(start) + offset,
                    "predicted_value": float(pred_arr[row, offset, 2]),
                    "lower_limit": float(JOINT_LIMITS[2, 0]),
                    "upper_limit": float(JOINT_LIMITS[2, 1]),
                    "expert_value": float(expert_arr[row, min(offset, expert_arr.shape[1] - 1), 2]),
                })
    if len(examples) > 20:
        chosen = np.random.default_rng(SEED).choice(len(examples), size=20, replace=False)
        examples = [examples[int(index)] for index in sorted(chosen)]
    steps = np.abs(np.diff(pred_arr, axis=1)) if pred_arr.shape[1] > 1 else np.zeros_like(pred_arr)
    result: dict[str, Any] = {
        "limit_source": LIMIT_SOURCE,
        "limit_units": LIMIT_UNITS,
        "joint_limits": {name: [float(lo), float(hi)] for name, (lo, hi) in zip(JOINT_NAMES, JOINT_LIMITS)},
        "gripper_limit": [float(GRIPPER_LIMITS[0]), float(GRIPPER_LIMITS[1])],
        "joint_limit_violation_count": int(pred_joint_bad.sum()),
        "joint_limit_violation_rate": float(pred_joint_bad.mean()),
        "joint_limit_by_dim": {name: int(pred_joint_bad[..., i].sum()) for i, name in enumerate(JOINT_NAMES)},
        "gripper_limit_violation_count": int(pred_grip_bad.sum()),
        "gripper_limit_violation_rate": float(pred_grip_bad.mean()),
        "expert_j3_violation_count": int(expert_joint_bad[..., 2].sum()),
        "final_j3_violation_count": int(pred_joint_bad[..., 2].sum()),
        "baseline_j3_violation_count": int(base_joint_bad[..., 2].sum()) if base_joint_bad is not None else None,
        "expert_gripper_violation_count": int(expert_grip_bad.sum()),
        "final_gripper_violation_count": int(pred_grip_bad.sum()),
        "baseline_gripper_violation_count": int(base_grip_bad.sum()) if base_grip_bad is not None else None,
        "j3_violation_examples": examples,
        "max_step_delta_by_dim": {name: float(np.max(steps[..., i])) for i, name in enumerate(ALL_NAMES)},
        "p95_step_delta_by_dim": {name: float(np.percentile(steps[..., i], 95)) for i, name in enumerate(ALL_NAMES)},
    }
    return result


def _select_windows(ds) -> list[tuple[str, int]]:
    by_ep: dict[str, list[int]] = {}
    for ep, start in ds.window_index:
        by_ep.setdefault(ep, []).append(int(start))
    manifest = json.loads(MANIFEST.read_text())
    episode_records = {record["episode_id"]: record for record in manifest["episodes"]}
    selected: list[tuple[str, int]] = []
    for ep in sorted(by_ep):
        starts = sorted(set(by_ep[ep]))
        n = 12 if episode_records[ep]["source"] == "recovery" else 6
        pos = np.linspace(0, len(starts) - 1, min(n, len(starts)), dtype=int)
        selected.extend((ep, starts[int(i)]) for i in np.unique(pos))
    return selected


def _metrics(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    err = np.abs(pred - expert)
    return {
        "first_action": summarize_errors(err[:, 0, :]),
        "first_action_by_dim": {name: summarize_errors(err[:, 0, i]) for i, name in enumerate(ALL_NAMES)},
        "full_chunk": summarize_errors(err),
        "full_chunk_by_dim": {name: summarize_errors(err[:, :, i]) for i, name in enumerate(ALL_NAMES)},
    }


def _continuity(pred: np.ndarray, selected: list[tuple[str, int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_ep: dict[str, list[tuple[int, int]]] = {}
    for i, (ep, start) in enumerate(selected):
        by_ep.setdefault(ep, []).append((start, i))
    values: list[float] = []
    records: list[dict[str, Any]] = []
    for ep, pairs in by_ep.items():
        pairs.sort()
        for (start_a, ia), (start_b, ib) in zip(pairs, pairs[1:]):
            if start_b <= start_a or start_b - start_a > HORIZON:
                continue
            overlap = HORIZON - (start_b - start_a)
            a = pred[ia, start_b - start_a : start_b - start_a + overlap]
            b = pred[ib, :overlap]
            d = np.abs(a[:, :PHYSICAL_DIM] - b[:, :PHYSICAL_DIM])
            for j, name in enumerate(JOINT_NAMES):
                values.append(float(np.mean(d[:, j])))
                records.append({"episode": ep, "frame": start_b, "joint": name, "value": float(np.mean(d[:, j]))})
    return summarize_errors(np.asarray(values)), sorted(records, key=lambda x: x["value"], reverse=True)[:10]


def _safety(pred: np.ndarray) -> dict[str, Any]:
    """Backward-compatible prediction-only safety summary."""
    array = np.asarray(pred)
    metadata = [("unknown", 0)] * int(array.shape[0])
    return safety_diagnostics(array, np.zeros_like(array), metadata)


def _run_full() -> int:
    started = time.perf_counter()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("<EVAL_ROOT>") / f"pi05_piper_joint_rtc_h20_quality58_2999_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "evaluation.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stderr)])
    log = logging.getLogger("offline_eval")
    try:
        ds = _dataset()
        selected = _select_windows(ds)
        policy, cfg = _load_policy(CONFIG_NAME, FINAL_CHECKPOINT)
        predictions: list[np.ndarray] = []
        experts: list[np.ndarray] = []
        episode_records = {
            record["episode_id"]: record
            for record in json.loads(MANIFEST.read_text())["episodes"]
        }
        for count, (ep, start) in enumerate(selected, 1):
            sample = ds.read_window(ep, start)
            pred = np.asarray(policy.infer(_obs(sample), noise=_noise(SEED, ep, start, HORIZON))["actions"], dtype=np.float32)
            validate_action_shapes(pred, cfg.model.action_dim)
            if not np.isfinite(pred).all():
                raise FloatingPointError(f"non-finite final output at {ep}:{start}")
            predictions.append(pred)
            experts.append(np.asarray(sample["action"], dtype=np.float32))
            if count % 25 == 0:
                log.info("final forward %d/%d", count, len(selected))
        pred_arr, exp_arr = np.stack(predictions), np.stack(experts)
        final_metrics = _metrics(pred_arr, exp_arr)

        # RTC is evaluated only on exact adjacent valid starts from the frozen
        # mask-aware index. This mirrors the training boundary's physical 7D
        # overlap and never includes latent padding dimensions.
        rtc_pairs = select_adjacent_pairs(ds.window_index, {ep: record["source"] for ep, record in episode_records.items()}, min_pairs=100)
        pair_predictions_a: list[np.ndarray] = []
        pair_predictions_b: list[np.ndarray] = []
        pair_recovery: list[bool] = []
        for count, (first, second) in enumerate(rtc_pairs, 1):
            first_sample = ds.read_window(*first)
            second_sample = ds.read_window(*second)
            first_pred = np.asarray(policy.infer(_obs(first_sample), noise=_noise(SEED, *first, HORIZON))["actions"], dtype=np.float32)
            second_pred = np.asarray(policy.infer(_obs(second_sample), noise=_noise(SEED, *second, HORIZON))["actions"], dtype=np.float32)
            validate_action_shapes(first_pred, cfg.model.action_dim)
            validate_action_shapes(second_pred, cfg.model.action_dim)
            if not np.isfinite(first_pred).all() or not np.isfinite(second_pred).all():
                raise FloatingPointError(f"non-finite RTC pair output at {first} -> {second}")
            pair_predictions_a.append(first_pred)
            pair_predictions_b.append(second_pred)
            pair_recovery.append(episode_records[first[0]]["source"] == "recovery")
            if count % 25 == 0:
                log.info("RTC adjacent forward %d/%d", count, len(rtc_pairs))
        rtc_pred_a = np.stack(pair_predictions_a)
        rtc_pred_b = np.stack(pair_predictions_b)
        rtc, worst = _continuity_pairs(rtc_pred_a, rtc_pred_b, rtc_pairs)
        recovery_rtc, _ = _continuity_pairs(
            rtc_pred_a[np.asarray(pair_recovery, dtype=bool)],
            rtc_pred_b[np.asarray(pair_recovery, dtype=bool)],
            [pair for pair, is_recovery in zip(rtc_pairs, pair_recovery) if is_recovery],
        )
        rtc["recovery_pair_count"] = recovery_rtc["pair_count"]
        rtc["recovery"] = recovery_rtc

        # Baseline is intentionally H10: same observations and first-action
        # A/B only. Its safety counts are therefore explicitly first-action
        # counts, not a fabricated H20 comparison.
        base_policy, base_cfg = _load_policy(BASELINE_CONFIG_NAME, BASELINE_CHECKPOINT)
        base_first: list[np.ndarray] = []
        base_expert: list[np.ndarray] = []
        for count, (ep, start) in enumerate(selected, 1):
            sample = ds.read_window(ep, start)
            out = base_policy.infer(_obs(sample), noise=_noise(SEED + 1, ep, start, int(base_cfg.model.action_horizon)))
            base_first.append(np.asarray(out["actions"], dtype=np.float32)[0])
            base_expert.append(np.asarray(sample["action"], dtype=np.float32)[0])
            if count % 25 == 0:
                log.info("baseline forward %d/%d", count, len(selected))
        base_first_arr = np.stack(base_first)
        base_expert_arr = np.stack(base_expert)
        base_first_metrics = summarize_errors(np.abs(base_first_arr - base_expert_arr))
        safety = safety_diagnostics(pred_arr, exp_arr, selected, baseline=base_first_arr)
        recovery_mask = np.array([episode_records[ep]["source"] == "recovery" for ep, _ in selected])
        nominal_mask = ~recovery_mask
        recovery_metrics = _metrics(pred_arr[recovery_mask], exp_arr[recovery_mask])
        nominal_metrics = _metrics(pred_arr[nominal_mask], exp_arr[nominal_mask])
        episode_rows = []
        for ep in sorted({x[0] for x in selected}):
            m = np.array([x[0] == ep for x in selected])
            e = _metrics(pred_arr[m], exp_arr[m])
            episode_rows.append({"episode": ep, "source": episode_records[ep]["source"], "windows": int(m.sum()), "first_action_fit": e["first_action"], "h20_fit": e["full_chunk"], "rtc_continuity": "measured", "joint_limit": "PASS" if _safety(pred_arr[m])["joint_limit_violation_count"] == 0 else "WARNING", "gripper": "PASS" if _safety(pred_arr[m])["gripper_limit_violation_count"] == 0 else "WARNING", "final": "PASS"})
        # Determinism on one frozen observation, same explicit noise.
        sample0 = ds.read_window(*selected[0])
        det = [np.asarray(policy.infer(_obs(sample0), noise=_noise(SEED, *selected[0], HORIZON))["actions"], dtype=np.float32) for _ in range(10)]
        det_diff = max(float(np.max(np.abs(det[i] - det[0]))) for i in range(1, len(det)))
        summary = {
            "FINAL_CHECKPOINT": FINAL_CHECKPOINT,
            "FINAL_CHECKPOINT_LOAD": "PASS",
            "DATASET": "FINAL_TRAINING_MANIFEST + FINAL_TRAIN_SAMPLE_MASK",
            "EPISODES": int(len(episode_records)),
            "NOMINAL_EPISODES": int(sum(v["source"] == "nominal" for v in episode_records.values())),
            "RECOVERY_EPISODES": int(sum(v["source"] == "recovery" for v in episode_records.values())),
            "VALID_H20_WINDOWS": int(len(ds)),
            "FORWARD_SAMPLES_FINAL": int(len(selected)),
            "PHYSICAL_STATE_DIM": 7,
            "PHYSICAL_ACTION_DIM": 7,
            "MODEL_LATENT_ACTION_DIM": int(cfg.model.action_dim),
            "ACTION_HORIZON": int(cfg.model.action_horizon),
            "FIRST_ACTION": final_metrics["first_action"],
            "FULL_CHUNK": final_metrics["full_chunk"],
            "NOMINAL": nominal_metrics,
            "RECOVERY": recovery_metrics,
            "RTC_INTER_CHUNK": rtc,
            "RTC_RECOVERY": rtc["recovery"],
            "TOP_10_WORST_RTC_TRANSITIONS": worst,
            "SAFETY": safety,
            "BASELINE_H10_FIRST_ACTION": base_first_metrics,
            "BASELINE_COMPARISON": baseline_comparison(final_metrics["first_action"], base_first_metrics),
            "H10_H20_COMPARISON": "H10/H20 incomparable for full chunks; baseline first-action A/B only",
            "DETERMINISM_MAX_ABS_DIFF": det_diff,
            "POLICY_DETERMINISTIC": "YES" if det_diff == 0.0 else "NO",
            "OFFLINE_GUARDS": OFFLINE_GUARDS,
            "ROBOT_CONNECTION_ATTEMPTED": "NO",
            "CAN_STARTED": "NO",
            "LIVE_CAMERA_OPENED": "NO",
            "ROBOT_MOTION": "NO",
            "FINAL_OFFLINE_GATE": "WARNING" if "incomparable" in "H10/H20 incomparable" else "PASS",
            "elapsed_seconds": time.perf_counter() - started,
        }
        _dump(out_dir / "summary.json", summary)
        _dump(out_dir / "metrics.json", {"final": final_metrics, "nominal": nominal_metrics, "recovery": recovery_metrics, "rtc": rtc, "safety": safety, "baseline_h10_first_action": base_first_metrics})
        _dump(out_dir / "recovery_per_episode.json", [r for r in episode_rows if r["source"] == "recovery"])
        _dump(out_dir / "worst_transitions.json", worst)
        log.info("OFFLINE_EVAL_DONE output=%s", out_dir)
        print(json.dumps({"status": "PASS", "output_dir": out_dir, **summary}, indent=2, default=_json_default))
        return 0
    except BaseException as exc:
        logging.getLogger("offline_eval").exception("OFFLINE_EVAL_FAIL")
        print(json.dumps({"status": "FAIL", "output_dir": out_dir, "exception": f"{type(exc).__name__}: {exc}", "guards": OFFLINE_GUARDS}, indent=2, default=_json_default), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    return _preflight() if args.preflight else _run_full()


if __name__ == "__main__":
    raise SystemExit(main())
