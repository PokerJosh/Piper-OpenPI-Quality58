"""Compute independent norm stats over every frozen Quality58 H20 window.

This intentionally bypasses the generic norm helper's drop_last=True loader so
that the final partial batch is included. The source dataset is still created
through the production ``create_torch_dataset`` dispatch and the exact
repack/data transforms used by the Quality58 config.
"""

import json
from pathlib import Path

import numpy as np
import torch
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as config_lib
import openpi.training.data_loader as data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {key: value for key, value in data.items() if not np.issubdtype(np.asarray(value).dtype, np.str_)}


def _collate(items: list[dict]) -> dict:
    return data_loader.jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values], axis=0), *items)


def main(
    config_name: str = "pi05_piper_joint_rtc_h20_quality58_finetune",
    batch_size: int = 16,
    expected_windows: int = 25336,
):
    config = config_lib.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if len(dataset) != expected_windows:
        raise RuntimeError(f"frozen Quality58 dataset length is {len(dataset)}, expected {expected_windows}")

    transformed = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    loader = torch.utils.data.DataLoader(
        transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=_collate,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    seen = 0
    batch_count = 0
    for batch in tqdm.tqdm(loader, total=(len(dataset) + batch_size - 1) // batch_size, desc="Quality58 norm"):
        batch_count += 1
        current = int(np.asarray(batch["state"]).shape[0])
        seen += current
        for key, running in stats.items():
            running.update(np.asarray(batch[key]))

    if seen != len(dataset):
        raise RuntimeError(f"normalization consumed {seen} windows, expected {len(dataset)}")

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}
    for key, value in norm_stats.items():
        for name in ("mean", "std", "q01", "q99"):
            array = getattr(value, name)
            if array is None or not np.all(np.isfinite(array)):
                raise RuntimeError(f"non-finite {key}.{name} in recomputed norm stats")
        if value.mean.shape != (7,) or value.std.shape != (7,):
            raise RuntimeError(f"expected 7D {key} stats, got {value.mean.shape} and {value.std.shape}")

    output_path = config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
    provenance = {
        "config_name": config_name,
        "manifest_path": data_config.mask_aware_manifest_path,
        "mask_path": data_config.mask_aware_mask_path,
        "dataset_class": type(dataset).__name__,
        "action_horizon": config.model.action_horizon,
        "action_dim": config.model.action_dim,
        "windows_seen": seen,
        "batch_size": batch_size,
        "batches_seen": batch_count,
        "drop_last": False,
        "transforms": [type(transform).__name__ for transform in (*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs)],
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "norm_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"NORM_STATS_WRITTEN={output_path / 'norm_stats.json'}")
    print(f"NORM_WINDOWS_SEEN={seen}")
    print(f"NORM_BATCHES_SEEN={batch_count}")
    print("NORM_RECOMPUTED_FROM_FINAL_MASK=YES")


if __name__ == "__main__":
    tyro.cli(main)
