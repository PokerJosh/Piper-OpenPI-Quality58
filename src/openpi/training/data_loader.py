from collections.abc import Iterator, Sequence
import io
import json
import logging
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import typing
from typing import ClassVar, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow.parquet as parquet
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

_JOINT_ACTION_KEYS = ("J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "G.pos")
_DEFAULT_JOINT_PROMPT = "pick up the battery and place it into the target location"


class _RecoveryCompatibilityUnpickler(pickle.Unpickler):
    """Loads old NumPy pickles without changing the process import table."""

    _MODULE_ALIASES: ClassVar[dict[str, str]] = {
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }

    def find_class(self, module: str, name: str):
        return super().find_class(self._MODULE_ALIASES.get(module, module), name)


def load_recovery_pickle_read_only(path: str | Path):
    """Read one recovery frame using a compatibility-only, read-only unpickler."""
    with Path(path).open("rb") as handle:
        return _RecoveryCompatibilityUnpickler(io.BytesIO(handle.read())).load()


def _expand_ranges(ranges: Sequence[Sequence[int]]) -> tuple[int, ...]:
    values: list[int] = []
    for item in ranges:
        if len(item) != 2:
            raise ValueError(f"frozen mask range must contain two integers: {item}")
        start, end = (int(item[0]), int(item[1]))
        if start > end:
            raise ValueError(f"frozen mask range is descending: {item}")
        values.extend(range(start, end + 1))
    return tuple(values)


def _episode_directory(root: Path, source_episode_id: str) -> Path:
    episode_id = str(source_episode_id)
    if not episode_id.startswith("episode_"):
        episode_id = f"episode_{episode_id}"
    return root / episode_id


def _as_frame_array(values, *, dtype=np.float32) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _canonical_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D camera frame, got {image.shape}")
    if image.shape[-1] == 3 and image.shape[0] != 3:
        image = np.transpose(image, (2, 0, 1))
    if image.shape[0] != 3:
        raise ValueError(f"expected a 3-channel camera frame, got {image.shape}")
    return image


class _NominalEpisodeReader:
    def __init__(self, root: Path, source_episode_id: str):
        self._root = root
        self._source_episode_id = str(source_episode_id)
        episode_number = int(self._source_episode_id.removeprefix("episode_"))
        self._parquet_path = root / "data" / "chunk-000" / f"episode_{episode_number:06d}.parquet"
        if not self._parquet_path.is_file():
            raise FileNotFoundError(f"nominal episode parquet not found: {self._parquet_path}")
        table = parquet.read_table(self._parquet_path, columns=["observation.state", "action", "timestamp"])
        self._state = _as_frame_array(table["observation.state"].to_pylist())
        self._action = _as_frame_array(table["action"].to_pylist())
        self._timestamps = _as_frame_array(table["timestamp"].to_pylist(), dtype=np.float64)
        if self._state.shape != (len(self._action), 7) or self._action.shape != (len(self._action), 7):
            raise ValueError(
                f"nominal episode must contain 7D state/actions, got {self._state.shape} and {self._action.shape}"
            )
        if self._timestamps.shape != (len(self._action),):
            raise ValueError(f"nominal timestamps must be 1D, got {self._timestamps.shape}")

    def frame_count(self) -> int:
        return len(self._action)

    def read(self, start: int, horizon: int, prompt: str) -> dict:
        end = start + horizon
        if start < 0 or end > self.frame_count():
            raise IndexError(f"H20 window [{start}:{end}] crosses nominal episode boundary")
        images = {}
        for camera in ("top", "wrist"):
            key = f"observation.images.{camera}"
            path = self._root / "videos" / "chunk-000" / key / f"{self._parquet_path.stem}.mp4"
            if not path.is_file():
                raise FileNotFoundError(f"nominal camera video not found: {path}")
            frame = lerobot_dataset.decode_video_frames(
                path,
                [float(self._timestamps[start])],
                tolerance_s=1e-4,
                backend=lerobot_dataset.get_safe_default_codec(),
            )
            images[key] = _canonical_image(frame.squeeze(0).cpu().numpy())
        return {
            "observation.state": self._state[start].copy(),
            "action": self._action[start:end].copy(),
            "observation.images.top": images["observation.images.top"],
            "observation.images.wrist": images["observation.images.wrist"],
            "prompt": prompt,
        }


class _RecoveryEpisodeReader:
    def __init__(self, root: Path, source_episode_id: str):
        self._episode_path = _episode_directory(root, source_episode_id)
        self._frame_paths = sorted(
            self._episode_path.glob("frame_*.pkl"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if not self._frame_paths:
            raise FileNotFoundError(f"recovery episode has no pickle frames: {self._episode_path}")

    def frame_count(self) -> int:
        return len(self._frame_paths)

    def read(self, start: int, horizon: int, prompt: str) -> dict:
        end = start + horizon
        if start < 0 or end > self.frame_count():
            raise IndexError(f"H20 window [{start}:{end}] crosses recovery episode boundary")
        frames = tuple(load_recovery_pickle_read_only(path) for path in self._frame_paths[start:end])
        first = frames[0]
        action = np.asarray(
            [[float(frame["action"][key]) for key in _JOINT_ACTION_KEYS] for frame in frames],
            dtype=np.float32,
        )
        state = np.asarray(
            [float(first["observation"][key]) for key in _JOINT_ACTION_KEYS],
            dtype=np.float32,
        )
        return {
            "observation.state": state,
            "action": action,
            "observation.images.top": _canonical_image(first["observation"]["top"]),
            "observation.images.wrist": _canonical_image(first["observation"]["wrist"]),
            "prompt": prompt,
        }


class MaskAwareJointRTCDataset:
    """Read-only H20 dataset backed by the frozen manifest and sample mask."""

    def __init__(
        self,
        manifest_path: str | Path,
        mask_path: str | Path,
        *,
        action_horizon: int = 20,
        seed: int | None = None,
        nominal_root: str | Path | None = None,
        recovery_root: str | Path | None = None,
        prompt: str = _DEFAULT_JOINT_PROMPT,
    ):
        if action_horizon != 20:
            raise ValueError("Joint RTC adapter requires H20 action_horizon=20")
        self.action_horizon = action_horizon
        self._manifest = json.loads(Path(manifest_path).read_text())
        self._mask = json.loads(Path(mask_path).read_text())
        if self._manifest.get("action_horizon") != 20 or self._mask.get("action_horizon") != 20:
            raise ValueError("frozen manifest and mask must both declare H20")
        manifest_episodes = {record["episode_id"]: record for record in self._manifest["episodes"]}
        mask_episodes = self._mask["episodes"]
        if set(manifest_episodes) != set(mask_episodes):
            raise ValueError("frozen manifest and mask episode IDs do not match")
        if "TOTAL_FINAL_EPISODES" in self._manifest:
            expected_sources = {
                "nominal": int(self._manifest["FINAL_NOMINAL_COUNT"]),
                "recovery": int(self._manifest["FINAL_RECOVERY_COUNT"]),
            }
            source_counts = {
                source: sum(record["source"] == source for record in manifest_episodes.values())
                for source in expected_sources
            }
            if (
                len(manifest_episodes) != int(self._manifest["TOTAL_FINAL_EPISODES"])
                or source_counts != expected_sources
            ):
                raise ValueError(
                    f"frozen selection counts mismatch: episodes={len(manifest_episodes)}, sources={source_counts}"
                )
        self._episode_records = manifest_episodes
        self._prompt = prompt
        self._nominal_root = Path(nominal_root) if nominal_root is not None else None
        self._recovery_root = Path(recovery_root) if recovery_root is not None else None
        self._readers: dict[str, _NominalEpisodeReader | _RecoveryEpisodeReader] = {}

        windows: list[tuple[str, int]] = []
        for episode_id, record in manifest_episodes.items():
            mask_record = mask_episodes[episode_id]
            starts = tuple(int(start) for start in mask_record["allowed_start_indices"])
            if starts != _expand_ranges(mask_record["allowed_start_ranges"]):
                raise ValueError(f"frozen mask ranges do not match explicit starts for {episode_id}")
            frame_count = int(record["frame_count"])
            if any(start < 0 or start + self.action_horizon > frame_count for start in starts):
                raise ValueError(f"frozen mask contains an out-of-bounds H20 start for {episode_id}")
            if int(record.get("allowed_start_count", len(starts))) != len(starts):
                raise ValueError(f"frozen manifest count mismatch for {episode_id}")
            windows.extend((episode_id, start) for start in starts)
        expected_windows = self._manifest.get("FINAL_VALID_H20_WINDOWS")
        if expected_windows is not None and len(windows) != int(expected_windows):
            raise ValueError(f"frozen H20 window count mismatch: {len(windows)} != {expected_windows}")
        if seed is not None:
            random.Random(seed).shuffle(windows)
        self.window_index = tuple(windows)
        self._window_set = frozenset(windows)

    def __len__(self) -> int:
        return len(self.window_index)

    def read_window(self, episode_id: str, start: int) -> dict:
        """Read one frozen H20 window, rejecting starts absent from the mask."""
        episode_id = str(episode_id)
        start = int(start)
        if (episode_id, start) not in self._window_set:
            raise IndexError(f"requested window ({episode_id!r}, {start}) is outside the frozen mask")
        return self._reader(episode_id).read(start, self.action_horizon, self._prompt)

    def _reader(self, episode_id: str):
        reader = self._readers.get(episode_id)
        if reader is not None:
            return reader
        record = self._episode_records[episode_id]
        source = record["source"]
        if source == "nominal":
            root = self._nominal_root or Path(record["dataset_root"])
            reader = _NominalEpisodeReader(root, record["source_episode_id"])
        elif source == "recovery":
            root = self._recovery_root or Path(record["dataset_root"])
            reader = _RecoveryEpisodeReader(root, record["source_episode_id"])
        else:
            raise ValueError(f"unsupported frozen source type: {source}")
        if reader.frame_count() != int(record["frame_count"]):
            raise ValueError(
                f"source frame count mismatch for {episode_id}: {reader.frame_count()} != {record['frame_count']}"
            )
        self._readers[episode_id] = reader
        return reader

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        if index < 0 or index >= len(self.window_index):
            raise IndexError(f"dataset index {index} is outside the frozen mask")
        episode_id, start = self.window_index[index]
        return self.read_window(episode_id, start)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    if data_config.mask_aware_manifest_path is not None or data_config.mask_aware_mask_path is not None:
        if data_config.mask_aware_manifest_path is None or data_config.mask_aware_mask_path is None:
            raise ValueError("mask-aware dataset requires both manifest and mask paths")
        if data_config.repo_id == "fake":
            raise ValueError("mask-aware dataset cannot use repo_id='fake'")
        return MaskAwareJointRTCDataset(
            data_config.mask_aware_manifest_path,
            data_config.mask_aware_mask_path,
            action_horizon=action_horizon,
            seed=data_config.mask_aware_seed,
            nominal_root=data_config.mask_aware_nominal_root,
            recovery_root=data_config.mask_aware_recovery_root,
            prompt=data_config.mask_aware_prompt or _DEFAULT_JOINT_PROMPT,
        )

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
