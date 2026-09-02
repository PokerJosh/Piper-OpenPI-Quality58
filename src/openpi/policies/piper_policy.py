"""Piper single-arm policy adapter.

Maps a Piper LeRobot dataset (and the inference-time robot inputs) to the pi0.5
model input space. The dataset uses:
- observation.images.top    -> base_0_rgb
- observation.images.wrist  -> left_wrist_0_rgb
- right_wrist_0_rgb is not present on this robot: zero image + False mask
- state: [J1..J6 in degrees, gripper in meters]
- actions: [J1..J6 absolute joint targets in degrees, gripper absolute target in meters]

NO unit conversion is performed here. The joint/gripper values are passed
through in their native units (degrees / meters); the `Normalize` model
transform (using stats computed from this same dataset via
scripts/compute_norm_stats.py) maps them into the model's expected range.
Because training and inference share these transforms, the mapping is
symmetric. The DeltaActions / AbsoluteActions transforms (mask = first 6
joints delta, gripper absolute) are applied in the data config, not here.
"""

import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms


def make_piper_example() -> dict:
    """Random input example for tests."""
    return {
        "state": np.zeros((7,)),
        "images": {
            "top": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick up the battery and place it into the target location",
    }


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Expected inputs:
    - images: dict with keys "top" and/or "wrist", values [C, H, W] (uint8 or float)
    - state: [7] (J1..J6 degrees, gripper meters)
    - actions: [action_horizon, 7] (only during training)
    - prompt: str (optional)
    """

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        base_image = _convert_image(in_images["top"])

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        if "wrist" in in_images:
            images["left_wrist_0_rgb"] = _convert_image(in_images["wrist"])
            image_masks["left_wrist_0_rgb"] = np.True_
        else:
            # Missing camera: zero image + False mask. Do NOT copy another camera.
            images["left_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["left_wrist_0_rgb"] = np.False_

        # This robot has no right wrist camera: always zero + masked out.
        images["right_wrist_0_rgb"] = np.zeros_like(base_image)
        image_masks["right_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperTCPAnchoredActions(transforms.DataTransformFn):
    """Convert absolute TCP target chunks to deltas anchored to the sample state."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        state = np.asarray(data["state"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)
        if state.shape != (7,) or actions.ndim != 2 or actions.shape[-1] != 7:
            raise ValueError(
                f"Expected state shape (7,) and actions shape (horizon, 7), got "
                f"{state.shape} and {actions.shape}"
            )

        # Dataset target is [position, rotvec, absolute gripper]. Convert each
        # future target to p_target[k] - p_state[t] and LogSO3(R_target[k] @
        # R_state[t].T). Rotvec subtraction is intentionally never used.
        anchored = np.empty_like(actions)
        anchored[:, :3] = actions[:, :3] - state[None, :3]
        target_rotation = Rotation.from_rotvec(actions[:, 3:6])
        state_rotation = Rotation.from_rotvec(state[3:6])
        anchored[:, 3:6] = (target_rotation * state_rotation.inv()).as_rotvec()
        anchored[:, 6] = actions[:, 6]

        output = dict(data)
        output["actions"] = anchored.astype(np.float32)
        return output


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Model outputs -> Piper 7D absolute actions (after AbsoluteActions inverse delta)."""

    def __call__(self, data: dict) -> dict:
        # The model predicts padded 32-dim actions; only the first 7 are physical.
        return {"actions": np.asarray(data["actions"][:, :7], dtype=np.float32)}


def _convert_image(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # [C, H, W] -> [H, W, C]
    return einops.rearrange(img, "c h w -> h w c")
