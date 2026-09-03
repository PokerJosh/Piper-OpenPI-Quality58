# Third-Party Notice

This archive combines project-specific Piper/OpenPI work with interfaces from upstream projects. It does not claim ownership of upstream code or hardware SDKs.

## Upstream and external components

- **OpenPI / Physical Intelligence**: model, transform, training, and policy interfaces. The upstream reference revision is recorded in `provenance/source_revisions.txt`; the selective archive is not a full upstream distribution.
- **LeRobot**: dataset, camera, robot, teleoperator, and recording interfaces used by the Piper plugins and conversion path.
- **Piper SDK**: external CAN-facing hardware API used at the adapter boundary. The archive preserves the project adapter and usage contract, not a claim over the SDK.
- **Feetech/SCServo SDK**: serial servo packet and group read/write support used by the leader kit.
- **Intel RealSense SDK**: wrist-camera integration at the LeRobot camera boundary.
- **ROS2/RViz and Piper description packages**: optional visualization/teleoperation integration referenced by templates; their full workspaces are not included.

Upstream license files already present in the archive remain authoritative. Preserve copyright headers and comply with each upstream license when using any file.

## Project-specific additions

The project-specific layer includes the Piper robot configuration and SDK adapter, Feetech leader mapping and calibration tools, guarded recording/review/conversion utilities, Piper policy transforms, mask-aware Quality58 H20 loading, RTC training helpers, normalization metadata, offline evaluation, Direct Joint safety core, guarded runner, focused tests, and final status documentation.

## Revisions

- Main OpenPI workspace snapshot: `c4c3b57277862c90a54173562e582f3cf1d4c2c0`.
- Direct Joint deployment worktree snapshot: `18cf73215927e95b960e98562cebdc8040c2effe`.
- Upstream OpenPI reference is retained in the existing provenance record.
- The hardware/recording source directory was not a Git repository; selected source paths are listed in the provenance record instead of inventing a source commit.
