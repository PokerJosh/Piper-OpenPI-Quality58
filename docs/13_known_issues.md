# 13 — Known Issues and Closure

## STALE_DATA_HOLD

The final real run stopped before policy command delivery because a cold-start JAX inference made the first observation/target exceed the runtime freshness age. The first-target wait was set to `FIRST_TARGET_TIMEOUT_SECONDS=20.0`, while the formal runtime gate remained `STALE_AFTER_SECONDS=0.50`. The resulting `STALE_DATA_HOLD` is the final recorded state.

A later idea was:

`cold-start warmup inference → discard warmup result → reacquire fresh camera → reacquire fresh CAN feedback → enter runtime`

Status: `PROPOSED_NOT_IMPLEMENTED`. It is not present as a completed fix, and the project is closed rather than continuing to repair it.

## Scope boundaries

The archive does not claim a successful task, benchmark success rate, or completed real deployment. Earlier TCP/ACT experiments are historical context and are not silently promoted to the final Joint Quality58 result. Real data and weights remain external.

If a selective-snapshot test cannot import omitted upstream packages, report `REQUIRES_UPSTREAM_OPENPI_TREE`; do not copy the entire upstream repository merely to make an archive test importable.

## Selective-snapshot validation

The archive can compile its Python files and run the bounded recording/review/guard tests without hardware. Some preserved deployment tests expect a concrete local camera identity or omitted upstream packages; those tests are not evidence of a production regression in this sanitized snapshot. Report them as `REQUIRES_UPSTREAM_OPENPI_TREE` or hardware-specific rather than substituting real identifiers.
