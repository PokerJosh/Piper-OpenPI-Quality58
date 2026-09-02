# Third-Party Notice

This archive is based on the upstream OpenPI project and its LeRobot-related
training/data interfaces. The upstream license and Gemma license notice are
preserved in `LICENSE` and `LICENSE_GEMMA.txt`.

## Upstream provenance

- Upstream project: OpenPI by Physical Intelligence.
- Upstream reference: the source repository's configured `origin/main` at the
  revision recorded in `provenance/source_revisions.txt`.
- The archive is not a replacement for the upstream source distribution.
- Upstream copyrights and license terms remain applicable.

## Project additions and modifications

The project-specific portion includes the Piper Joint policy transforms,
Quality58 H20 mask-aware dataset adapter, training-time RTC helpers/configuration,
normalization metadata, offline evaluator, Direct Joint safety core, guarded
Joint2999 runner, camera manifest validation, focused tests, and final project
documentation.

The files in `provenance/deployment_snapshot/` preserve the final deployment
worktree versions of shared configuration files that differed from the main
worktree. They are included for auditability and are not a claim of upstream
ownership.
