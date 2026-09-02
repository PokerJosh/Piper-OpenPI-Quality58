#!/usr/bin/env python3
"""Hard validation and metadata freeze for the Joint RTC Quality58 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


QA_ROOT = Path("<DATA_QA_ROOT>/joint_rtc_expert_salvage_20260901T161500Z")
NOMINAL_ROOT = Path("<NOMINAL_DATASET_ROOT>")
RECOVERY_ROOT = Path("<RECOVERY_DATASET_ROOT>")
EXPECTED_EPISODES = 58
EXPECTED_WINDOWS = 25336
HORIZON = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(path: Path) -> str:
    return sha256_file(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _range_indices(ranges: list[list[int]]) -> list[int]:
    values: list[int] = []
    for item in ranges:
        _require(len(item) == 2, f"range must have two values: {item}")
        start, end = (int(item[0]), int(item[1]))
        _require(start <= end, f"descending range: {item}")
        values.extend(range(start, end + 1))
    return values


def _source_fingerprint(root: Path, selected_episode_ids: list[str], source_type: str) -> dict[str, Any]:
    meta_paths = sorted((root / "meta").glob("*"))
    meta_hashes = {str(path.relative_to(root)): sha256_file(path) for path in meta_paths if path.is_file()}
    selected_files: list[dict[str, Any]] = []
    for episode_id in selected_episode_ids:
        episode = int(episode_id)
        if source_type == "nominal":
            parquet = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
            video_paths = [
                root / "videos" / "chunk-000" / key / f"episode_{episode:06d}.mp4"
                for key in ("observation.images.top", "observation.images.wrist")
            ]
        else:
            parquet = root / f"episode_{episode:06d}" / "frame_000000.pkl"
            video_paths = []
        paths = [parquet, *video_paths]
        for path in paths:
            _require(path.exists(), f"missing source file: {path}")
            stat = path.stat()
            selected_files.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "sha256": sha256_file(path) if path.suffix == ".parquet" else None,
                }
            )
    payload = {"root": str(root), "meta_sha256": meta_hashes, "selected_files": selected_files}
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    return payload


def validate(qa_root: Path = QA_ROOT) -> dict[str, Any]:
    manifest_path = qa_root / "FINAL_TRAINING_MANIFEST.json"
    mask_path = qa_root / "FINAL_TRAIN_SAMPLE_MASK.json"
    review_path = qa_root / "visual_review_results.json"
    review_summary_path = qa_root / "visual_review_summary.json"
    train_mask_path = qa_root / "TRAIN_SAMPLE_MASK.json"
    review_manifest_path = qa_root / "review_manifest.json"
    for path in (
        manifest_path,
        mask_path,
        review_path,
        review_summary_path,
        train_mask_path,
        review_manifest_path,
    ):
        _require(path.is_file(), f"missing QA artifact: {path}")

    manifest = load_json(manifest_path)
    mask = load_json(mask_path)
    reviews = load_json(review_path)
    review_summary = load_json(review_summary_path)
    review_manifest = load_json(review_manifest_path)

    episodes = manifest["episodes"]
    mask_episodes = mask["episodes"]
    _require(manifest["action_horizon"] == HORIZON, "manifest horizon mismatch")
    _require(mask["action_horizon"] == HORIZON, "mask horizon mismatch")
    _require(manifest["fps"] == 30 and mask["fps"] == 30, "fps mismatch")
    _require(len(episodes) == EXPECTED_EPISODES, f"expected {EXPECTED_EPISODES} episodes")
    _require(len({episode["episode_id"] for episode in episodes}) == EXPECTED_EPISODES, "duplicate episode ids")
    _require(len(mask_episodes) == EXPECTED_EPISODES, "mask episode count mismatch")

    nominal = [episode for episode in episodes if episode["source"] == "nominal"]
    recovery = [episode for episode in episodes if episode["source"] == "recovery"]
    _require(len(nominal) == 47, f"nominal count mismatch: {len(nominal)}")
    _require(len(recovery) == 11, f"recovery count mismatch: {len(recovery)}")
    _require(set(mask_episodes) == {episode["episode_id"] for episode in episodes}, "manifest/mask key mismatch")

    review_statuses = {key: value["review_status"] for key, value in reviews.items()}
    _require(len(review_statuses) == 58, "review result count mismatch")
    _require(all(status in {"PASS_A", "PASS_B", "PASS"} for status in review_statuses.values()), "non-pass review found")
    _require(
        Counter(value for key, value in review_statuses.items() if key.startswith("nominal_"))
        == Counter({"PASS_A": 22, "PASS_B": 25}),
        "nominal visual review counts mismatch",
    )
    _require(
        Counter(value for key, value in review_statuses.items() if key.startswith("recovery_")) == Counter({"PASS": 11}),
        "recovery visual review counts mismatch",
    )
    _require(review_summary["nominal_visual_pass_count"] == 47, "nominal visual pass summary mismatch")
    _require(review_summary["recovery_visual_pass_count"] == 11, "recovery visual pass summary mismatch")
    _require(review_summary["fail_count"] == 0 and review_summary["ambiguous_count"] == 0, "failed visual review exists")

    quality_counts = {}
    source_windows = {}
    all_windows = 0
    for source_name, source_episodes in (("nominal", nominal), ("recovery", recovery)):
        source_quality = Counter()
        source_window_count = 0
        for episode in source_episodes:
            episode_id = episode["episode_id"]
            _require(review_statuses.get(episode_id) in {"PASS_A", "PASS_B", "PASS"}, f"episode not visually passed: {episode_id}")
            _require(episode["mask_key"] == episode_id, f"mask key mismatch: {episode_id}")
            _require(episode_id in mask_episodes, f"missing mask: {episode_id}")
            entry = mask_episodes[episode_id]
            indices = _range_indices(entry["allowed_start_ranges"])
            _require(indices == sorted(set(indices)), f"overlapping/unsorted ranges: {episode_id}")
            _require(indices == [int(x) for x in entry["allowed_start_indices"]], f"expanded ranges mismatch: {episode_id}")
            frame_count = int(episode["frame_count"])
            legal_max = frame_count - HORIZON
            _require(all(0 <= start <= legal_max for start in indices), f"out-of-bounds H20 start: {episode_id}")
            _require(len(indices) == int(entry["valid_h20_windows"]) == int(episode["valid_h20_windows"]), f"window count mismatch: {episode_id}")
            _require(int(entry["total_h20_windows"]) == frame_count - HORIZON + 1, f"total H20 count mismatch: {episode_id}")
            _require(episode["quality_class"] in {"CLEAN", "SALVAGEABLE"}, f"bad quality class: {episode_id}")
            _require(entry["source"] == source_name, f"mask source mismatch: {episode_id}")
            if source_name == "nominal":
                _require(entry["visual_review_status"] in {"PASS_A", "PASS_B"}, f"nominal mask review mismatch: {episode_id}")
            else:
                _require(entry["visual_review_status"] == "PASS", f"recovery mask review mismatch: {episode_id}")
            source_quality[episode["quality_class"]] += 1
            source_window_count += len(indices)
            all_windows += len(indices)
        quality_counts[source_name] = dict(source_quality)
        source_windows[source_name] = source_window_count

    _require(all_windows == EXPECTED_WINDOWS, f"expected {EXPECTED_WINDOWS} legal windows, got {all_windows}")
    _require(source_windows == {"nominal": 19889, "recovery": 5447}, f"source window totals mismatch: {source_windows}")

    diversity = manifest["diversity_recomputed"]
    _require(diversity["coverage"]["formatted"] == "7/7 (100%)", "diversity coverage mismatch")
    nominal_clusters = {int(episode["diversity_cluster"]) for episode in nominal}
    _require(nominal_clusters == set(range(7)), f"diversity cluster coverage mismatch: {nominal_clusters}")
    _require(manifest["DIVERSITY_GATE"] == "PASS", "manifest diversity gate is not PASS")
    _require(manifest["QUALITY_GATE"] == "PASS", "manifest quality gate is not PASS")
    _require(manifest["DATA_QUANTITY_GATE"] == "PASS", "manifest data quantity gate is not PASS")

    source_episode_ids = {
        "nominal": [episode["source_episode_id"] for episode in nominal],
        "recovery": [episode["source_episode_id"] for episode in recovery],
    }
    _require(len(set(source_episode_ids["nominal"])) == 47, "duplicate nominal source episode")
    _require(len(set(source_episode_ids["recovery"])) == 11, "duplicate recovery source episode")
    _require(all(episode["dataset_root"] == str(NOMINAL_ROOT) for episode in nominal), "nominal source root mismatch")
    _require(all(episode["dataset_root"] == str(RECOVERY_ROOT) for episode in recovery), "recovery source root mismatch")
    _require(review_manifest["total_episodes"] == 58, "review manifest total mismatch")

    result = {
        "validated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "qa_root": str(qa_root),
        "final_manifest": str(manifest_path),
        "final_mask": str(mask_path),
        "counts": {
            "final_nominal_count": len(nominal),
            "final_recovery_count": len(recovery),
            "total_final_episodes": len(episodes),
            "nominal_clean_count": quality_counts["nominal"].get("CLEAN", 0),
            "nominal_salvageable_count": quality_counts["nominal"].get("SALVAGEABLE", 0),
            "recovery_clean_count": quality_counts["recovery"].get("CLEAN", 0),
            "recovery_salvageable_count": quality_counts["recovery"].get("SALVAGEABLE", 0),
            "final_valid_h20_windows": all_windows,
        },
        "source_windows": source_windows,
        "diversity": {
            "cluster_count": 7,
            "coverage": "7/7",
            "cluster_ids": sorted(nominal_clusters),
            "feature_components": diversity["feature_components"],
        },
        "gates": {
            "visual_success_nominal": 47,
            "quality_gate": "PASS",
            "diversity_gate": "PASS",
            "data_quantity_gate": "PASS",
            "dataset_ready_for_rtc_training": "YES",
            "more_data_needed": "NO",
        },
        "input_sha256": {
            name: sha256_json(qa_root / name)
            for name in (
                "FINAL_TRAINING_MANIFEST.json",
                "FINAL_TRAIN_SAMPLE_MASK.json",
                "visual_review_results.json",
                "visual_review_summary.json",
                "TRAIN_SAMPLE_MASK.json",
                "review_manifest.json",
            )
        },
        "source_fingerprints": {
            "nominal": _source_fingerprint(NOMINAL_ROOT, source_episode_ids["nominal"], "nominal"),
            "recovery": _source_fingerprint(RECOVERY_ROOT, source_episode_ids["recovery"], "recovery"),
        },
        "read_only_contract": {
            "source_data_modified": "NO",
            "training_started": "NO",
            "norm_recomputed": "NO",
            "real_robot_executed": "NO",
            "can_write": "NO",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", type=Path, default=QA_ROOT)
    parser.add_argument("--write-freeze", action="store_true")
    args = parser.parse_args()
    result = validate(args.qa_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.write_freeze:
        output = args.qa_root / "RTC_QUALITY58_DATA_FREEZE.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"FREEZE_WRITTEN={output}")
    print("VALIDATION=PASS")


if __name__ == "__main__":
    main()
