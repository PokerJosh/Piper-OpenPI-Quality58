#!/usr/bin/env python3
"""Pure config-selection + recovery-path guard logic for custom_record_final.py.

Kept import-light (no lerobot chain) so tests run in any env. custom_record_final.py
imports these functions; they must not trigger lerobot/robot/teleop imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

DEFAULT_REVIEW_CONFIG = "record_review_config.yaml"
RECOVERY_CONFIG_MARKER = "recovery"
RECOVERY_RAW_MARKER = "piper_openpi_v21_recovery_v1"
RECOVERY_REVIEWED_MARKER = "piper_openpi_v21_recovery_v1_reviewed"
LEGACY_MARKER = "piper_demo_new_20260821_184850"
INSERT_3MM_CONFIG_MARKER = "insert_3mm"
INSERT_3MM_RAW_MARKER = "piper_insert_3mm_v1"
INSERT_3MM_REVIEWED_MARKER = "piper_insert_3mm_v1_reviewed"


def is_recovery_review_config(path: Optional[Union[str, Path]]) -> bool:
    """True only when the review-config path is an explicit Recovery config.

    Identified by the "recovery" marker in the file name. The default config and
    any custom non-recovery YAML return False (guard stays OFF for them).
    """
    if not path:
        return False
    return RECOVERY_CONFIG_MARKER in Path(path).name


def is_insert_3mm_review_config(path: Optional[Union[str, Path]]) -> bool:
    """True only when the review-config file name carries the insert_3mm marker."""
    if not path:
        return False
    return INSERT_3MM_CONFIG_MARKER in Path(path).name


def insert_3mm_guard(
    raw_root: Union[str, Path],
    reviewed_root: Union[str, Path],
) -> Tuple[bool, bool, str]:
    """Insert-3mm configs may access ONLY the insert roots and never recovery/legacy roots."""
    raw = str(raw_root)
    rev = str(reviewed_root)
    if INSERT_3MM_RAW_MARKER not in raw:
        return True, False, f"REFUSE: insert_3mm raw_root missing {INSERT_3MM_RAW_MARKER!r}: {raw}"
    if INSERT_3MM_REVIEWED_MARKER not in rev:
        return True, False, f"REFUSE: insert_3mm reviewed_root missing {INSERT_3MM_REVIEWED_MARKER!r}: {rev}"
    if RECOVERY_RAW_MARKER in raw or RECOVERY_REVIEWED_MARKER in rev:
        return True, False, f"REFUSE: insert_3mm config must NOT point at recovery roots: {raw} / {rev}"
    if LEGACY_MARKER in raw or LEGACY_MARKER in rev:
        return True, False, f"REFUSE: insert_3mm config must NOT point at legacy roots: {raw} / {rev}"
    return True, True, "PASS: insert_3mm roots OK"


def recovery_guard(
    review_config_path: Optional[Union[str, Path]],
    raw_root: Union[str, Path],
    reviewed_root: Union[str, Path],
) -> Tuple[bool, bool, str]:
    """Return (guard_on, ok, reason).

    guard_on is True ONLY for an explicit Recovery or Insert-3mm review config;
    for the default config or a custom non-recovery YAML it is False and ok is
    True (never REFUSE).  When guard_on, ok requires the roots to carry the
    matching dataset marker and never the recovery/legacy markers (REFUSE before
    any mkdir / robot command / recording).
    """
    if is_insert_3mm_review_config(review_config_path):
        return insert_3mm_guard(raw_root, reviewed_root)
    if not is_recovery_review_config(review_config_path):
        return False, True, "GUARD_OFF: not a Recovery review config"

    raw = str(raw_root)
    rev = str(reviewed_root)
    if RECOVERY_RAW_MARKER not in raw:
        return True, False, f"REFUSE: raw_root missing {RECOVERY_RAW_MARKER!r}: {raw}"
    if RECOVERY_REVIEWED_MARKER not in rev:
        return True, False, f"REFUSE: reviewed_root missing {RECOVERY_REVIEWED_MARKER!r}: {rev}"
    if LEGACY_MARKER in raw:
        return True, False, f"REFUSE: raw_root still points at legacy {LEGACY_MARKER!r}: {raw}"
    if LEGACY_MARKER in rev:
        return True, False, f"REFUSE: reviewed_root still points at legacy {LEGACY_MARKER!r}: {rev}"
    return True, True, "PASS: recovery roots OK"
