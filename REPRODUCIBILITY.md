# Reproducibility

This archive intentionally contains code and metadata, not the source data or
model weights.

## Required external inputs

Provide these values in the local environment or in a private, local copy of
the relevant configuration before running an offline operation:

- `<OPENPI_ROOT>`: a compatible upstream OpenPI checkout.
- `<CHECKPOINT_ROOT>`: the separately retained checkpoint root, if a load-only
  or offline evaluator is approved.
- `<NOMINAL_DATASET_ROOT>`: the nominal Piper Joint dataset.
- `<RECOVERY_DATASET_ROOT>`: the reviewed recovery Piper Joint dataset.
- `<DATA_QA_ROOT>`: the final training manifest and sample mask directory.
- `<TOP_CAMERA_DEVICE_PATH>` and a local wrist-camera serial: only for a
  separately approved hardware operation. Hardware is outside this archive.

The archived files use placeholders instead of machine-specific values. They
are reference snapshots and are not intended to silently discover devices or
local files.

## Frozen training facts

- 47 nominal episodes and 11 recovery episodes.
- 58 total episodes.
- 25,336 legal H20 windows.
- H20 action horizon.
- 7 physical action dimensions, with 32 model action dimensions.
- Training-time RTC maximum delay 4.
- 3,000 training steps; final logical checkpoint step 2,999.

The metadata-only examples under `examples/` preserve selection and window
membership information without copying episode data.

## Safe validation boundary

Allowed archive validation is static compilation and small CPU/offline unit
tests only. Do not run camera probes, CAN operations, robot commands, model
training, or long GPU evaluations as part of archive verification.

Because this is a selective snapshot, tests that import omitted upstream
modules require `REQUIRES_UPSTREAM_OPENPI_TREE`. Replace placeholders only in a
working copy or through local environment/configuration; do not commit local
hardware identifiers or private paths back into this archive.
