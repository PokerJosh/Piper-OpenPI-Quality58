# 00 — Overview

The project path is a staged pipeline, not one monolithic command:

`hardware setup → CAN/SDK → calibration → leader/follower teleoperation → cameras → recording → review/convert → nominal/recovery selection → π0.5 Quality58 RTC H20 training → offline evaluation → guarded Direct Joint deployment → safety-gated real run`.

The final pipeline uses one Piper arm with two image views (top and wrist), seven physical outputs, a 32-dimensional π0.5 model action representation, and H20 action chunks. Earlier ACT and TCP experiments are retained only where they explain the data path; they are not the final result.

All commands in this archive are templates. Archive validation is static/CPU-only and does not open cameras, configure CAN, enable an arm, train, or run inference on hardware.
