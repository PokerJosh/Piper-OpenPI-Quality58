# 09 — Offline Evaluation

The focused evaluator is `scripts/eval_joint_pi05_rtc_h20_quality58_final_offline.py`; its pure tests are under `scripts/test_eval_joint_pi05_rtc_h20_quality58_final_offline.py`. Supporting Quality58 QA and normalization tools are in `dataset_tools/qa/` and `scripts/`.

A complete approved offline evaluation should check:

- preflight/config/schema;
- first-action metrics;
- full H20 chunk metrics;
- RTC prefix continuity;
- nominal/recovery breakdown;
- deterministic repeated evaluation;
- hard-limit and raw-reject diagnostics;
- inference latency and schema dimensions.

The final offline observation was a positive J3 policy overshoot tendency. This is a diagnostic tendency, not a success-rate claim and not proof that every action overshoots. It is one reason hard-limit and raw-output rejection remain mandatory.

The archived evaluator requires a compatible upstream tree and separately retained checkpoint/data roots. Archive validation runs only pure functions/unit tests and does not load weights, use a GPU, or access cameras/CAN.
