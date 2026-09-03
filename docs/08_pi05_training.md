# 08 — π0.5 Quality58 Training

The final config is `pi05_piper_joint_rtc_h20_quality58_finetune` in the archived OpenPI training config. It uses:

- physical action dimension: 7;
- model/latent action dimension: 32;
- action horizon: H20;
- training-time RTC enabled with maximum delay 4;
- `LeRobotPiperJointRTCQuality58DataConfig`;
- `MaskAwareJointRTCDataset` as the frozen-manifest/sample-mask entry point.

The data loader reads nominal LeRobot episodes and recovery frame pickles through separate readers, never crosses an episode boundary, and accepts only starts present in the frozen mask. J1–J6 use the joint delta/absolute transform while gripper remains absolute. The policy transform maps top/wrist images and seven-dimensional state to the model input; normalization uses the small archived stats artifact generated from the same selection.

Training RTC conditions the committed physical prefix while retaining padded model dimensions for the 32-wide representation. `training_rtc`, `training_rtc_max_delay`, and `training_rtc_action_dim=7` are model fields. Prefix steps are excluded from the physical RTC loss; padding/unpadding must not turn latent padding dimensions into physical commands.

The recorded run used 3,000 steps and logical checkpoint 2,999. The checkpoint reference is `pi05_piper_joint_rtc_h20_quality58_3k_20260902_135854/2999`; weights are not included. For a local resume/load, provide `<CHECKPOINT_ROOT>` and verify config, norm stats, action dimensions, and checkpoint provenance before use. Do not commit checkpoint trees or dataset roots.
