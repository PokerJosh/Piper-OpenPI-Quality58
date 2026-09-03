# 07 — Dataset Processing

The project processing chain is:

`raw episode → schema/timestamp/camera QA → conservative trim/review → visual/dry replay → LeRobot conversion → nominal/recovery selection → manifest and sample mask → H20 window validation → normalization statistics`.

`recording/episode_tools.py` validates frame numbering, seven-dimensional state/action keys, finite values, camera schemas, timestamps, and duplicate/stale-frame warnings. `recording/review_episode.py` and `review_all_episodes.py` write approved output to a separate root. `recording/replay_episode.py` provides visual and offline trajectory checks; real replay is explicitly gated and is not part of validation.

`recording/convert_to_lerobot.py` converts embedded raw frames into a LeRobot schema with fixed joint ordering and top/wrist image keys. The `dataset_tools/conversion/` scripts document the v2.1-compatible conversion used by the OpenPI data path; their source and destination roots are placeholders only.

The final Quality58 selection is 47 nominal plus 11 recovery episodes, 58 total, with 25,336 valid H20 windows. `dataset_tools/qa/prepare_piper_joint_recovery_mix.py` describes immutable source-to-mix construction; `validate_joint_rtc_quality58.py` freezes counts, review status, and window membership. Real manifests may contain paths, so only metadata-only examples are included.

No episode payload, parquet, pickle frame, video, or raw capture is in this repository.
