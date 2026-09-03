# Command Reference

All commands below are templates. Replace placeholders in a local working copy after an independent safety review. Archive validation does not execute hardware, CAN, training, or long GPU evaluation.

## Environment

```bash
cd <PIPER_PROJECT_ROOT>
# Activate the locally pinned Python 3.12 environment.
OPENPI_ROOT=<OPENPI_ROOT> bash hardware/piper_upper_pc/install_piper_upper_pc.sh --root <PIPER_PROJECT_ROOT>
```

## CAN and hardware read checks

```bash
ip link | grep can
sudo ip link set <CAN_INTERFACE> up type can bitrate 1000000
ls <LEADER_SERIAL_PORT>
```

The commands above can alter hardware state and are operator-only templates.

## Calibration

```bash
python calibration/scripts/feetech_ping.py --port <LEADER_SERIAL_PORT> --scan 1-7
python calibration/scripts/feetech_servo_zero.py --port <LEADER_SERIAL_PORT> --verify
python calibration/scripts/leader_servo_mapping.py --ticks 2047,2047,2047,2047,2047,2047,2047
```

EEPROM-writing calibration requires a separate review and private generated result; do not replace the example JSON in this repository with real values.

## Teleoperation and camera check

```bash
bash recording/record_piper_teleop.sh --dry-run
python scripts/verify_piper_joint2999_cameras.py --manifest configs/piper_joint2999_camera.yaml
```

For a real recording, provide `<TOP_CAMERA_SERIAL>`, `<WRIST_CAMERA_SERIAL>`, `<TOP_CAMERA_DEVICE_PATH>`, `<CAN_INTERFACE>`, and `<LEADER_SERIAL_PORT>` through a local config. The camera verifier may open devices; do not run it as archive validation.

## Record and review episodes

```bash
python recording/custom_record_final.py --camera-config configs/piper_recording.example.yaml --dry-check-config
python recording/review_all_episodes.py --input-root <DATASET_ROOT>/raw --output-root <DATASET_ROOT>/reviewed
python recording/replay_episode.py --episode <DATASET_ROOT>/reviewed/episode_XXXXXX --mode robot
python recording/convert_to_lerobot.py --input-root <DATASET_ROOT>/reviewed --repo-id local/piper_demo_reviewed
```

Keep reset motion outside the episode. Real replay requires its explicit safety gates and is not a normal archive step.

## Dataset QA and selection

```bash
python dataset_tools/conversion/piper_convert_v21.py
python dataset_tools/qa/prepare_piper_joint_recovery_mix.py --original <NOMINAL_DATASET_ROOT> --recovery <RECOVERY_DATASET_ROOT> --destination <DATASET_ROOT>/joint_mix
python dataset_tools/qa/validate_joint_rtc_quality58.py --qa-root <DATA_QA_ROOT>
python dataset_tools/qa/compute_joint_rtc_quality58_norm.py
```

These templates operate on external data; no real dataset is included.

## Training and resume

```bash
python scripts/smoke_joint_rtc_h20_quality58.py --config pi05_piper_joint_rtc_h20_quality58_finetune
python scripts/train.py --config pi05_piper_joint_rtc_h20_quality58_finetune
python scripts/train.py --config pi05_piper_joint_rtc_h20_quality58_finetune --resume <CHECKPOINT_ROOT>
```

The recorded final facts are 3,000 steps and logical step 2,999. Do not use the checkpoint reference as a local path.

## Offline evaluation and deployment

```bash
python scripts/eval_joint_pi05_rtc_h20_quality58_final_offline.py --checkpoint <CHECKPOINT_ROOT> --dataset-root <DATASET_ROOT>
python scripts/pi05_joint2999_rollout.py --dry-run
python scripts/pi05_joint2999_rollout.py --preflight
```

A real runner requires independent review, explicit acknowledgement, local camera/feedback inputs, and a write adapter. This archive does not run it. See `docs/10_direct_joint_deployment.md`, `docs/11_real_robot_run.md`, and `docs/12_safety.md`.
