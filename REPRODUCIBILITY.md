# Reproducibility

This repository is a selective engineering archive, not a replacement for the upstream OpenPI or LeRobot source tree. It preserves project-specific code, templates, tests, procedures, and final status while omitting real payloads.

## External inputs

Supply these values only in a local working copy or environment:

- `<OPENPI_ROOT>`: compatible upstream OpenPI checkout.
- `<PIPER_PROJECT_ROOT>`: local checkout containing the hardware/recording integration.
- `<CAN_INTERFACE>`: local Piper CAN interface.
- `<LEADER_SERIAL_PORT>`: local Feetech leader serial port.
- `<TOP_CAMERA_DEVICE_PATH>` and `<TOP_CAMERA_SERIAL>`: local top-camera identity.
- `<WRIST_CAMERA_SERIAL>`: local wrist-camera identity.
- `<DATASET_ROOT>`: local raw/reviewed/conversion data root.
- `<NOMINAL_DATASET_ROOT>` and `<RECOVERY_DATASET_ROOT>`: local source datasets.
- `<CHECKPOINT_ROOT>`: separately retained checkpoint root.
- `<DATA_QA_ROOT>`: local manifest and mask directory.

Do not commit replacements for these placeholders.

## Quality58 freeze

The final training selection was 47 nominal episodes plus 11 recovery episodes, 58 total, yielding 25,336 valid H20 windows. The model used physical action dimension 7, latent/model action dimension 32, and action horizon 20. Training ran for 3,000 steps, with logical final checkpoint step 2,999.

The frozen manifest and sample mask are metadata concepts only in this repository. The episode files, videos, and source dataset roots are not included.

## Validation boundary

Permitted archive checks are Python compilation and bounded CPU/offline tests. Do not run camera probes, CAN operations, robot commands, training, or long GPU evaluation as part of archive verification. Some selective-snapshot tests require the omitted upstream tree and should be reported as `REQUIRES_UPSTREAM_OPENPI_TREE` rather than solved by copying the entire upstream project.
