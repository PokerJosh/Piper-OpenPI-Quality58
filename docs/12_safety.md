# 12 — Safety Design

The final safety design is fail-closed and must not be weakened for an archive or future run:

- physical hard joint limits;
- gripper limits;
- finite/shape validation;
- raw policy output rejection;
- safety abort;
- executor re-check immediately before handoff;
- maximum 1 degree/tick for J1–J6;
- speed and timing caps;
- camera freshness checking;
- feedback freshness checking;
- target freshness checking;
- driver fault checking;
- explicit arm enable and write-adapter gates.

The physical output is seven values, but the policy model carries 32 dimensions. Only the physical seven may reach a Piper command boundary. A rejected raw target must never be silently clamped into a command without preserving the rejection policy.

Offline evaluation observed a positive J3 overshoot tendency. Hard limits and raw-output rejection are therefore essential diagnostics and protection, not optional performance knobs.

The archive itself performs no CAN write, camera probe, arm enable, or real rollout.
