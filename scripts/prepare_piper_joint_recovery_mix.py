#!/usr/bin/env python3
"""Build an immutable 92+11 Joint-space LeRobot v2 mix.

The input datasets are never changed. Original episodes remain 0..91; reviewed
recovery episodes are copied as 92..102 after verifying the shared 7D Joint
schema. Recovery parquet metadata is rewritten only in the destination so its
episode and global frame indices match the new mix. Videos are hardlinked, not
re-encoded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


JOINT_NAMES = ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "G.pos"]
VIDEO_KEYS = ("observation.images.top", "observation.images.wrist")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_episodes(root: Path) -> list[dict[str, Any]]:
    path = root / "meta/episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"episode indices are not contiguous in {path}")
    return rows


def _dataset_summary(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = (root / "meta/info.json", root / "meta/stats.json", root / "meta/tasks.jsonl")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing dataset metadata: {missing}")
    info = _load_json(root / "meta/info.json")
    episodes = _load_episodes(root)
    frames = sum(int(row["length"]) for row in episodes)
    if int(info["total_episodes"]) != len(episodes) or int(info["total_frames"]) != frames:
        raise ValueError(f"metadata totals disagree in {root}")
    if float(info["fps"]) != 30:
        raise ValueError(f"expected 30fps in {root}, got {info['fps']}")
    return info, episodes


def _validate_joint_schema(original_info: dict[str, Any], recovery_info: dict[str, Any]) -> None:
    for label, info in (("original", original_info), ("recovery", recovery_info)):
        features = info.get("features", {})
        for key in ("observation.state", "action"):
            feature = features.get(key, {})
            if feature.get("shape") != [7] or feature.get("names") != JOINT_NAMES:
                raise ValueError(f"{label} {key} is not the required 7D Joint schema")
        for key in VIDEO_KEYS:
            if key not in features:
                raise ValueError(f"{label} is missing {key}")
    for key in ("observation.state", "action"):
        if original_info["features"][key] != recovery_info["features"][key]:
            raise ValueError(f"schema mismatch for {key}")


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _require_episode_files(root: Path, episode: int) -> None:
    stem = f"episode_{episode:06d}"
    required = [root / "data/chunk-000" / f"{stem}.parquet"]
    required.extend(root / "videos/chunk-000" / key / f"{stem}.mp4" for key in VIDEO_KEYS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing episode assets: {missing}")


def _copy_recovery_parquet(source: Path, destination: Path, source_episode: int, target_episode: int, index_offset: int) -> None:
    import pandas as pd

    frame = pd.read_parquet(source)
    required_columns = {"observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"recovery parquet missing columns: {required_columns - set(frame.columns)}")
    if not (frame["episode_index"] == source_episode).all():
        raise ValueError(f"recovery parquet episode index mismatch: {source}")
    if frame["frame_index"].tolist() != list(range(len(frame))):
        raise ValueError(f"recovery parquet frame index mismatch: {source}")
    frame = frame.copy()
    frame["episode_index"] = target_episode
    frame["index"] = range(index_offset, index_offset + len(frame))
    frame.to_parquet(destination, index=False)


def build_mix(original: Path, recovery: Path, destination: Path, *, validate_parquet: bool = True) -> dict[str, Any]:
    """Create a new mix and return its verified mapping report."""
    original, recovery, destination = original.resolve(), recovery.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    if destination in (original, recovery):
        raise ValueError("destination must be distinct from both input datasets")

    original_info, original_episodes = _dataset_summary(original)
    recovery_info, recovery_episodes = _dataset_summary(recovery)
    _validate_joint_schema(original_info, recovery_info)
    if (original / "meta/tasks.jsonl").read_bytes() != (recovery / "meta/tasks.jsonl").read_bytes():
        raise ValueError("original and recovery task metadata differ")

    destination.mkdir(parents=True)
    try:
        (destination / "meta").mkdir()
        shutil.copy2(original / "meta/tasks.jsonl", destination / "meta/tasks.jsonl")
        info = dict(original_info)
        info["total_episodes"] = len(original_episodes) + len(recovery_episodes)
        info["total_frames"] = sum(int(row["length"]) for row in original_episodes + recovery_episodes)
        (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        # OpenPI uses independently computed norm stats. This stale-compatible
        # LeRobot metadata is retained only for layout compatibility.
        shutil.copy2(original / "meta/stats.json", destination / "meta/stats.json")

        output_rows: list[dict[str, Any]] = []
        for row in original_episodes:
            source_episode = int(row["episode_index"])
            _require_episode_files(original, source_episode)
            stem = f"episode_{source_episode:06d}"
            _link(original / "data/chunk-000" / f"{stem}.parquet", destination / "data/chunk-000" / f"{stem}.parquet")
            for key in VIDEO_KEYS:
                _link(
                    original / "videos/chunk-000" / key / f"{stem}.mp4",
                    destination / "videos/chunk-000" / key / f"{stem}.mp4",
                )
            output_rows.append(dict(row))

        offset = int(original_info["total_frames"])
        recovery_mapping: dict[int, int] = {}
        for row in recovery_episodes:
            source_episode = int(row["episode_index"])
            target_episode = len(original_episodes) + source_episode
            recovery_mapping[source_episode] = target_episode
            _require_episode_files(recovery, source_episode)
            source_stem, target_stem = f"episode_{source_episode:06d}", f"episode_{target_episode:06d}"
            target_parquet = destination / "data/chunk-000" / f"{target_stem}.parquet"
            if validate_parquet:
                _copy_recovery_parquet(
                    recovery / "data/chunk-000" / f"{source_stem}.parquet",
                    target_parquet,
                    source_episode,
                    target_episode,
                    offset,
                )
            else:
                _link(recovery / "data/chunk-000" / f"{source_stem}.parquet", target_parquet)
            for key in VIDEO_KEYS:
                _link(
                    recovery / "videos/chunk-000" / key / f"{source_stem}.mp4",
                    destination / "videos/chunk-000" / key / f"{target_stem}.mp4",
                )
            output_row = dict(row)
            output_row["episode_index"] = target_episode
            output_rows.append(output_row)
            offset += int(row["length"])

        (destination / "meta/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in output_rows) + "\n")
        report = {
            "original_episodes": len(original_episodes),
            "original_frames": int(original_info["total_frames"]),
            "recovery_episodes": len(recovery_episodes),
            "recovery_frames": int(recovery_info["total_frames"]),
            "mix_episodes": len(output_rows),
            "mix_frames": int(info["total_frames"]),
            "recovery_mapping": recovery_mapping,
        }
        (destination / "joint_recovery_mix_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    except Exception:
        shutil.rmtree(destination)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = build_mix(args.original, args.recovery, args.destination)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
