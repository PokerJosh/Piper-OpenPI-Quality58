"""Convert the Piper LeRobot v3.0 dataset into a v2.1-compatible derived dataset.

The original dataset at
  ~/.cache/huggingface/lerobot/local/piper_demo_new_20260821_184850_v2
is left completely untouched. This script creates a NEW dataset at
  ~/.cache/huggingface/lerobot/local/piper_openpi_v21
in the LeRobot v2.x layout expected by the lerobot version pinned by openpi:
  - per-episode parquet files:  data/chunk-000/episode_XXXXXX.parquet
  - per-episode videos:         videos/chunk-000/<key>/episode_XXXXXX.mp4
  - meta/tasks.jsonl, meta/episodes.jsonl, meta/stats.json, meta/info.json

Video content is stream-copied (no re-encode) from the consolidated v3 files,
cut at each episode's exact [from_timestamp, to_timestamp].
"""

import json
import pathlib
import subprocess

import pandas as pd

from pathlib import Path
SRC = Path("<DATASET_ROOT>/local/piper_openpi_v21_recovery_v1_clean_11ep_v3raw")
DST = Path("<DATASET_ROOT>/local/piper_openpi_v21_recovery_v1_clean_11ep")

VIDEO_KEYS = ["observation.images.top", "observation.images.wrist"]

(DST / "meta").mkdir(parents=True, exist_ok=True)
(DST / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
for k in VIDEO_KEYS:
    (DST / "videos" / "chunk-000" / k).mkdir(parents=True, exist_ok=True)

episodes = pd.read_parquet(SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
data = pd.read_parquet(SRC / "data" / "chunk-000" / "file-000.parquet")

# --- meta/tasks.jsonl (keep original task string) ---
tasks = pd.read_parquet(SRC / "meta" / "tasks.parquet")
task_index = int(tasks["task_index"].iloc[0])
task_str = str(tasks.index[0])
with open(DST / "meta" / "tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": task_index, "task": task_str}) + "\n")

# --- per-episode data, videos, episodes.jsonl ---
with open(DST / "meta" / "episodes.jsonl", "w") as ef:
    for _, ep in episodes.iterrows():
        i = int(ep["episode_index"])
        length = int(ep["length"])
        sub = data[data["episode_index"] == i]
        assert len(sub) == length, (i, len(sub), length)
        sub.to_parquet(DST / "data" / "chunk-000" / f"episode_{i:06d}.parquet", index=False)
        for k in VIDEO_KEYS:
            file_index = int(ep[f"videos/{k}/file_index"])
            t_from = float(ep[f"videos/{k}/from_timestamp"])
            t_to = float(ep[f"videos/{k}/to_timestamp"])
            src_vid = SRC / "videos" / k / "chunk-000" / f"file-{file_index:03d}.mp4"
            dst_vid = DST / "videos" / "chunk-000" / k / f"episode_{i:06d}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-ss", f"{t_from:.6f}", "-to", f"{t_to:.6f}", "-i", str(src_vid),
                    "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst_vid),
                ],
                check=True,
            )
        ef.write(json.dumps({"episode_index": i, "tasks": [task_str], "length": length}) + "\n")
        print(f"episode {i} done ({length} frames)", flush=True)

# --- meta/info.json: v2.0 layout keys + original features/stats ---
info = json.loads((SRC / "meta" / "info.json").read_text())
info["codebase_version"] = "v2.0"
info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
info["chunks_size"] = 1000
info["total_episodes"] = len(episodes)
info["total_frames"] = len(data)
info["total_tasks"] = 1
info.pop("index_path", None)
info.pop("episode_manifest", None)
(DST / "meta" / "info.json").write_text(json.dumps(info, indent=2))

# stats.json: copy as-is (same schema in v2/v3)
stats = json.loads((SRC / "meta" / "stats.json").read_text())
(DST / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

print("DONE", DST)
