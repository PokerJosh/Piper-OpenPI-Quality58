import dataclasses
import json
from pathlib import Path
import pickle

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

_JOINT_ACTION_KEYS = ("J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "G.pos")


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def _write_episode_fixture(root: Path, episode_id: str, length: int) -> None:
    for frame in range(length):
        frame_data = {
            "observation": {
                **{key: float(frame) for key in _JOINT_ACTION_KEYS},
                "top": np.full((2, 2, 3), frame, dtype=np.uint8),
                "wrist": np.full((2, 2, 3), frame + 1, dtype=np.uint8),
            },
            "action": {key: float(frame + 100) for key in _JOINT_ACTION_KEYS},
        }
        (root / episode_id).mkdir(parents=True, exist_ok=True)
        with (root / episode_id / f"frame_{frame:06d}.pkl").open("wb") as handle:
            pickle.dump(frame_data, handle, protocol=4)


def _write_mask_fixture(
    root: Path,
    episodes: dict[str, list[int]],
    *,
    frame_count: int,
) -> tuple[Path, Path]:
    manifest_path = root / "manifest.json"
    mask_path = root / "mask.json"
    manifest_path.write_text(
        json.dumps(
            {
                "action_horizon": 20,
                "episodes": [
                    {
                        "episode_id": episode_id,
                        "source": "recovery",
                        "source_episode_id": episode_id,
                        "dataset_root": str(root / "episodes"),
                        "mask_key": episode_id,
                        "frame_count": frame_count,
                    }
                    for episode_id, starts in episodes.items()
                ],
            }
        )
    )
    mask_path.write_text(
        json.dumps(
            {
                "action_horizon": 20,
                "episodes": {
                    episode_id: {
                        "episode_id": episode_id,
                        "source": "recovery",
                        "source_episode_id": episode_id,
                        "allowed_start_indices": starts,
                        "allowed_start_ranges": [[start, start] for start in starts],
                    }
                    for episode_id, starts in episodes.items()
                },
            }
        )
    )
    return manifest_path, mask_path


def test_mask_dataset_samples_only_frozen_legal_windows(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    _write_episode_fixture(episodes_root, "episode_000000", 42)
    manifest_path, mask_path = _write_mask_fixture(tmp_path, {"episode_000000": [3, 7]}, frame_count=42)

    dataset = _data_loader.MaskAwareJointRTCDataset(
        manifest_path,
        mask_path,
        recovery_root=episodes_root,
    )

    assert len(dataset) == 2
    assert dataset.window_index == (("episode_000000", 3), ("episode_000000", 7))
    assert np.array_equal(dataset[0]["action"], np.arange(103, 123, dtype=np.float32)[:, None].repeat(7, axis=1))


def test_mask_dataset_rejects_boundary_crossing_and_unmasked_starts(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    _write_episode_fixture(episodes_root, "episode_000000", 25)
    manifest_path, mask_path = _write_mask_fixture(tmp_path, {"episode_000000": [5]}, frame_count=25)
    dataset = _data_loader.MaskAwareJointRTCDataset(
        manifest_path,
        mask_path,
        recovery_root=episodes_root,
    )

    with pytest.raises(IndexError, match="frozen mask"):
        dataset[1]
    with pytest.raises(IndexError, match="frozen mask"):
        dataset.read_window("episode_000000", 4)
    with pytest.raises(ValueError, match="H20"):
        _data_loader.MaskAwareJointRTCDataset(
            manifest_path,
            mask_path,
            recovery_root=episodes_root,
            action_horizon=21,
        )


def test_mask_dataset_rejects_inconsistent_frozen_ranges(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    _write_episode_fixture(episodes_root, "episode_000000", 30)
    manifest_path, mask_path = _write_mask_fixture(tmp_path, {"episode_000000": [0]}, frame_count=30)
    mask = json.loads(mask_path.read_text())
    mask["episodes"]["episode_000000"]["allowed_start_ranges"] = [[1, 1]]
    mask_path.write_text(json.dumps(mask))

    with pytest.raises(ValueError, match="ranges"):
        _data_loader.MaskAwareJointRTCDataset(
            manifest_path,
            mask_path,
            recovery_root=episodes_root,
        )


def test_recovery_pickle_loader_reads_numpy_compatibility_pickle_read_only(tmp_path: Path):
    source_root = tmp_path / "episodes"
    source_root.mkdir()
    frame_path = source_root / "frame.pkl"
    old_numpy_pickle = pickle.dumps(np.asarray([1], dtype=np.float32), protocol=0).replace(
        b"numpy.core.multiarray",
        b"numpy._core.multiarray",
    )
    frame_path.write_bytes(old_numpy_pickle)

    loaded = _data_loader.load_recovery_pickle_read_only(frame_path)

    assert np.array_equal(loaded, np.asarray([1], dtype=np.float32))
    assert frame_path.read_bytes() == old_numpy_pickle


def test_mask_dataset_seeded_order_is_deterministic(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    _write_episode_fixture(episodes_root, "episode_000000", 60)
    _write_episode_fixture(episodes_root, "episode_000001", 60)
    manifest_path, mask_path = _write_mask_fixture(
        tmp_path,
        {"episode_000000": [1, 2, 3], "episode_000001": [4, 5, 6]},
        frame_count=60,
    )

    first = _data_loader.MaskAwareJointRTCDataset(
        manifest_path,
        mask_path,
        seed=23,
        recovery_root=episodes_root,
    )
    second = _data_loader.MaskAwareJointRTCDataset(
        manifest_path,
        mask_path,
        seed=23,
        recovery_root=episodes_root,
    )
    different = _data_loader.MaskAwareJointRTCDataset(
        manifest_path,
        mask_path,
        seed=24,
        recovery_root=episodes_root,
    )

    assert first.window_index == second.window_index
    assert first.window_index != different.window_index


def test_mask_dataset_does_not_write_source_tree(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    _write_episode_fixture(episodes_root, "episode_000000", 30)
    manifest_path, mask_path = _write_mask_fixture(tmp_path, {"episode_000000": [0]}, frame_count=30)
    before = sorted((path.relative_to(episodes_root), path.stat().st_mtime_ns) for path in episodes_root.rglob("*"))

    dataset = _data_loader.MaskAwareJointRTCDataset(
        manifest_path,
        mask_path,
        recovery_root=episodes_root,
    )
    _ = dataset[0]

    after = sorted((path.relative_to(episodes_root), path.stat().st_mtime_ns) for path in episodes_root.rglob("*"))
    assert after == before


def test_quality58_config_uses_frozen_mask_as_the_only_torch_dataset_entrypoint(tmp_path: Path):
    episodes_root = tmp_path / "episodes"
    _write_episode_fixture(episodes_root, "episode_000000", 42)
    manifest_path, mask_path = _write_mask_fixture(tmp_path, {"episode_000000": [3, 7]}, frame_count=42)

    factory = _config.LeRobotPiperJointRTCQuality58DataConfig(
        manifest_path=str(manifest_path),
        mask_path=str(mask_path),
        recovery_root=str(episodes_root),
    )
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=20,
        training_rtc=True,
        training_rtc_action_dim=7,
    )
    data_config = factory.create(tmp_path / "assets", model_config)

    dataset = _data_loader.create_torch_dataset(data_config, action_horizon=20, model_config=model_config)

    assert isinstance(dataset, _data_loader.MaskAwareJointRTCDataset)
    assert len(dataset) == 2
    assert dataset.window_index == (("episode_000000", 3), ("episode_000000", 7))
