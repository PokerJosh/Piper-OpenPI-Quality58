import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_piper_joint_recovery_mix.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("joint_mix", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset(root: Path, lengths: list[int], label: str) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    for key in ("observation.images.top", "observation.images.wrist"):
        (root / "videos/chunk-000" / key).mkdir(parents=True)
    info = {
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "fps": 30,
        "features": {
            "observation.state": {"shape": [7], "names": ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "G.pos"]},
            "action": {"shape": [7], "names": ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "G.pos"]},
            "observation.images.top": {"dtype": "video"},
            "observation.images.wrist": {"dtype": "video"},
        },
    }
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text("{}")
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "pick and place"}\n')
    rows = []
    for index, length in enumerate(lengths):
        name = f"episode_{index:06d}"
        (root / "data/chunk-000" / f"{name}.parquet").write_text(f"{label}-data-{index}")
        for key in ("observation.images.top", "observation.images.wrist"):
            (root / "videos/chunk-000" / key / f"{name}.mp4").write_text(f"{label}-video-{index}")
        rows.append(json.dumps({"episode_index": index, "tasks": ["pick and place"], "length": length}))
    (root / "meta/episodes.jsonl").write_text("\n".join(rows) + "\n")


def test_build_mix_preserves_sources_and_offsets_recovery_episode_ids(tmp_path: Path):
    """Catches a merger that overwrites inputs or fails to map recovery after originals."""
    module = _load_module()
    original = tmp_path / "original"
    recovery = tmp_path / "recovery"
    mixed = tmp_path / "mixed"
    _write_dataset(original, [2, 3], "original")
    _write_dataset(recovery, [4], "recovery")

    report = module.build_mix(original, recovery, mixed, validate_parquet=False)

    assert report["mix_episodes"] == 3
    assert report["mix_frames"] == 9
    assert report["recovery_mapping"] == {0: 2}
    assert json.loads((mixed / "meta/episodes.jsonl").read_text().splitlines()[2])["episode_index"] == 2
    assert (original / "data/chunk-000/episode_000000.parquet").read_text() == "original-data-0"
    assert (recovery / "data/chunk-000/episode_000000.parquet").read_text() == "recovery-data-0"
