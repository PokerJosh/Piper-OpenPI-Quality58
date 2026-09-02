import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import openpi.training.config as training_config
from openpi.policies import piper_policy


def _rotvec(rotation: Rotation) -> np.ndarray:
    return rotation.as_rotvec()


def _delta(anchor_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    anchor_rotation = Rotation.from_rotvec(anchor_pose[3:6])
    target_rotation = Rotation.from_rotvec(target_pose[3:6])
    return np.concatenate(
        [
            target_pose[:3] - anchor_pose[:3],
            _rotvec(target_rotation * anchor_rotation.inv()),
            [target_pose[6]],
        ]
    )


def _synthetic_chunk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_state = np.array([0.10, 0.02, 0.05, 0.0, 0.0, 0.20, 0.03], dtype=np.float32)
    horizon = 10
    future_states = []
    absolute_targets = []
    for k in range(horizon):
        future_states.append(
            np.array(
                [
                    0.10 + 0.002 * k,
                    0.02 + 0.0005 * k,
                    0.05 - 0.0004 * k,
                    0.0,
                    0.0,
                    0.20 + 0.01 * k,
                    0.03,
                ],
                dtype=np.float32,
            )
        )
        absolute_targets.append(
            np.array(
                [
                    0.105 + 0.004 * k,
                    0.025 + 0.0008 * k,
                    0.052 - 0.0007 * k,
                    0.0,
                    0.0,
                    0.22 + 0.015 * k,
                    0.032,
                ],
                dtype=np.float32,
            )
        )
    future_states = np.stack(future_states)
    absolute_targets = np.stack(absolute_targets)
    v2_local_actions = np.stack([_delta(future_states[k], absolute_targets[k]) for k in range(horizon)])
    return anchor_state, absolute_targets, v2_local_actions


def test_v2_local_action_chunk_is_not_anchored_to_sample_state():
    """Documents the audited V2 bug: its future rows are anchored to state[t+k]."""
    anchor_state, absolute_targets, v2_local_actions = _synthetic_chunk()
    transformed = piper_policy.PiperInputs()(
        {
            "images": {"top": np.zeros((3, 2, 2), dtype=np.uint8)},
            "state": anchor_state,
            "actions": v2_local_actions,
        }
    )["actions"]

    # If V2 actions were already anchored to state[t], PiperInputs would not need
    # to alter them. Their first six dimensions therefore must differ from A.
    anchored_actions = np.stack([_delta(anchor_state, target) for target in absolute_targets])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(transformed[:, :6], anchored_actions[:, :6], atol=1e-9)


def test_v3_action_chunk_is_anchored_to_sample_state():
    anchor_state, absolute_targets, _ = _synthetic_chunk()
    config = training_config.get_config("pi05_piper_tcp_delta_v3_finetune")
    data_config = config.data.create(config.assets_dirs, config.model)

    data = {
        "images": {"top": np.zeros((3, 2, 2), dtype=np.uint8)},
        "state": anchor_state,
        "actions": absolute_targets,
    }
    for transform in data_config.data_transforms.inputs:
        data = transform(data)
    transformed = data["actions"]
    expected = np.stack([_delta(anchor_state, target) for target in absolute_targets])

    np.testing.assert_allclose(transformed, expected, rtol=0.0, atol=1e-9)
