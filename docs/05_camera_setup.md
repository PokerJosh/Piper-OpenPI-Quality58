# 05 — Camera Setup

The final observation uses two views:

- `top`: fixed/top RGB view.
- `wrist`: wrist-mounted RealSense RGB view.

The Piper policy maps them to π0.5 `base_0_rgb` and `left_wrist_0_rgb`; the unused right-wrist slot is masked rather than filled by copying another camera. Camera frames are converted to the expected channel/order and resolution by the policy input transform.

Use `configs/piper_recording.example.yaml` for recording and `configs/piper_joint2999_camera.yaml` for the final deployment shape. Supply `<TOP_CAMERA_DEVICE_PATH>`, `<TOP_CAMERA_SERIAL>`, and `<WRIST_CAMERA_SERIAL>` locally. No real camera serial or stable device identity is archived.

`verify_piper_joint2999_cameras.py` is a manifest/probe implementation. It validates role mapping, shape, timestamps, and freshness. The camera probe is a hardware operation and was not run during archive validation. Camera freshness is also checked at runtime; a valid image shape alone is not sufficient.
