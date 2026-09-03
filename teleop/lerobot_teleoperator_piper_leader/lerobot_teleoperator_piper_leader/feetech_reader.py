"""飞特主臂串口读 ticks（从 piper_leader_kit / ROS reader 抽取，无 ROS 依赖）。"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


class FeetechLeaderReader:
    SMS_STS_PRESENT_POSITION_L = None  # set after SDK import
    SMS_STS_TORQUE_ENABLE = None       # set after SDK import
    SMS_STS_PRESENT_VOLTAGE = None     # set after SDK import
    SMS_STS_MODEL_L = None             # set after SDK import
    SMS_STS_MODEL_H = None             # set after SDK import
    # Voltage-limit EPROM registers (1 byte each). Not named in this SDK's
    # sms_sts.py; addresses verified against lerobot feetech tables.py:
    #   "Max_Voltage_Limit": (14, 1)  /  "Min_Voltage_Limit": (15, 1)
    SMS_STS_MAX_VOLTAGE_LIMIT = 14
    SMS_STS_MIN_VOLTAGE_LIMIT = 15

    def __init__(
        self,
        port: str = "<LEADER_SERIAL_PORT>",
        baudrate: int = 1_000_000,
        servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
        scservo_sdk_path: str | Path | None = None,
        max_tick_jump: int = 120,
        hold_last_on_read_error: bool = True,
        require_all_servos: bool = True,
    ):
        self.port_name = port
        self.baudrate = baudrate
        self.servo_ids = list(servo_ids)
        self.scservo_sdk_path = Path(scservo_sdk_path).expanduser() if scservo_sdk_path else None
        self.max_tick_jump = max_tick_jump
        self.hold_last_on_read_error = hold_last_on_read_error
        self.require_all_servos = require_all_servos

        self._last_good_raw: list[int | None] = [None] * len(self.servo_ids)
        self.port_handler = None
        self.packet_handler = None
        self.group_reader = None
        self._comm_success = None
        self._comm_rx_corrupt = None
        self._comm_rx_timeout = None
        self._comm_tx_fail = None

        # Per-servo torque write state from the LAST enable/disable call. A bad
        # status-packet ack (COMM_RX_CORRUPT/TIMEOUT) does NOT prove the write
        # missed the servo (TX is transmitted first), so a failing id is
        # recorded as UNKNOWN, not as not-applied.
        self.torque_write_attempted: list[int] = []
        self.torque_write_confirmed: list[int] = []
        self.torque_write_unknown: list[int] = []
        self.torque_write_not_applied: list[int] = []

    @staticmethod
    def _resolve_sdk_path(configured: Path | None) -> Path:
        candidates: list[Path] = []
        if configured is not None:
            candidates.append(configured)
        import os

        if os.environ.get("SCSERVO_SDK_PATH"):
            candidates.append(Path(os.environ["SCSERVO_SDK_PATH"]).expanduser())
        candidates.append(Path("<PIPER_PROJECT_ROOT>/hardware/piper_leader_kit/sdk"))

        for path in candidates:
            if (path / "scservo_sdk" / "__init__.py").is_file():
                return path
        searched = ", ".join(str(p) for p in candidates)
        raise RuntimeError(f"找不到 scservo_sdk，已搜索: {searched}")

    def _import_sdk(self):
        sdk_path = self._resolve_sdk_path(self.scservo_sdk_path)
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        scservo_sdk = importlib.import_module("scservo_sdk")
        self.PortHandler = scservo_sdk.PortHandler
        self.SmsSts = scservo_sdk.sms_sts
        self.GroupSyncRead = scservo_sdk.GroupSyncRead
        type(self).SMS_STS_PRESENT_POSITION_L = scservo_sdk.SMS_STS_PRESENT_POSITION_L
        type(self).SMS_STS_TORQUE_ENABLE = scservo_sdk.SMS_STS_TORQUE_ENABLE
        type(self).SMS_STS_PRESENT_VOLTAGE = scservo_sdk.SMS_STS_PRESENT_VOLTAGE
        type(self).SMS_STS_MODEL_L = scservo_sdk.SMS_STS_MODEL_L
        type(self).SMS_STS_MODEL_H = scservo_sdk.SMS_STS_MODEL_H
        self._comm_success = scservo_sdk.COMM_SUCCESS
        self._comm_rx_corrupt = scservo_sdk.COMM_RX_CORRUPT
        self._comm_rx_timeout = scservo_sdk.COMM_RX_TIMEOUT
        self._comm_tx_fail = scservo_sdk.COMM_TX_FAIL

    def connect(self) -> None:
        self._import_sdk()
        self.port_handler = self.PortHandler(self.port_name)
        self.packet_handler = self.SmsSts(self.port_handler)
        self.group_reader = self.GroupSyncRead(
            self.packet_handler, self.SMS_STS_PRESENT_POSITION_L, 4
        )
        if not self.port_handler.openPort():
            raise RuntimeError(f"无法打开串口 {self.port_name}")
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"无法设置波特率 {self.baudrate}")

        logger.info(
            "Feetech 串口已打开: port=%s baud=%d ids=%s",
            self.port_name,
            self.baudrate,
            self.servo_ids,
        )

    def disconnect(self) -> None:
        if self.port_handler is not None and self.port_handler.is_open:
            self.port_handler.closePort()
        self.port_handler = None
        self.packet_handler = None
        self.group_reader = None

    @property
    def is_connected(self) -> bool:
        return self.port_handler is not None and self.port_handler.is_open

    def _filter_raw_tick(self, raw: int, index: int) -> int | None:
        if raw < 0:
            if self.hold_last_on_read_error and self._last_good_raw[index] is not None:
                return self._last_good_raw[index]
            return None

        last = self._last_good_raw[index]
        if last is not None and self.max_tick_jump > 0:
            delta = int(raw) - int(last)
            if delta > 2048:
                delta -= 4096
            elif delta < -2048:
                delta += 4096
            if abs(delta) > self.max_tick_jump:
                logger.debug(
                    "舵机 id=%s tick 跳变 %d 超过 max_tick_jump=%d，沿用上一帧",
                    self.servo_ids[index],
                    delta,
                    self.max_tick_jump,
                )
                return last

        self._last_good_raw[index] = int(raw)
        return int(raw)

    def read_raw_by_id(self) -> dict[int, int] | None:
        if not self.is_connected:
            raise ConnectionError("FeetechLeaderReader is not connected.")

        # 每次重新创建 GroupSyncRead 避免缓存问题
        group_reader = self.GroupSyncRead(
            self.packet_handler, self.SMS_STS_PRESENT_POSITION_L, 4
        )

        for servo_id in self.servo_ids:
            group_reader.addParam(servo_id)

        comm_result = group_reader.txRxPacket()
        raw_list = [-1] * len(self.servo_ids)

        if comm_result == self._comm_success:
            for index, servo_id in enumerate(self.servo_ids):
                available, servo_error = group_reader.isAvailable(
                    servo_id, self.SMS_STS_PRESENT_POSITION_L, 4
                )
                # 主臂舵机被动读取时 error=1 是正常的，只要数据可用就读取
                if available:
                    raw = group_reader.getData(
                        servo_id, self.SMS_STS_PRESENT_POSITION_L, 2
                    )
                    raw_list[index] = int(raw)

        filtered: dict[int, int] = {}
        for index, servo_id in enumerate(self.servo_ids):
            value = self._filter_raw_tick(raw_list[index], index)
            if value is None:
                if self.require_all_servos:
                    return None
                continue
            filtered[servo_id] = value

        if self.require_all_servos and len(filtered) != len(self.servo_ids):
            missing = [sid for sid in self.servo_ids if sid not in filtered]
            logger.warning(
                "Feetech 读数不完整，缺少 ID %s（原始=%s）。"
                "检查供电、ID、接线；或设 require_all_servos=false",
                missing,
                raw_list,
            )
            return None
        return filtered

    def read_raw_by_id_or_raise(self) -> dict[int, int]:
        """读 ticks；失败时抛出带诊断信息的异常。"""
        raw = self.read_raw_by_id()
        if raw is not None:
            return raw
        raise RuntimeError(
            f"无法从 {self.port_name} 读取 7 路舵机 ticks。"
            "请确认：USB 线连接、舵机上电、端口号正确（ls /dev/ttyACM*）、"
            "用户在 dialout 组、无其它程序占用串口（如 ROS reader 节点）。"
        )

    def read_ticks(self) -> dict[int, int] | None:
        """ActiveHandoff/bench bus protocol: same read as read_raw_by_id.

        Returns None (not raise) when a read is incomplete/stale so callers
        (mirror convergence, bench hold) can detect a silent comm failure.
        Single serial owner: reuses this reader's port/packet handlers.
        """
        return self.read_raw_by_id()

    # ------------------------------------------------------------------
    # ACTIVE-HANDOFF WRITE PATH (same serial owner as the read path —
    # NEVER open a second port handler for /dev/ttyACM0).
    # ------------------------------------------------------------------
    def _assert_connected(self):
        if not self.is_connected or self.packet_handler is None:
            raise ConnectionError("FeetechLeaderReader is not connected.")

    def _torque_write(self, value: int, verb: str, ids=None) -> None:
        """Sequential per-servo torque register write + status ack.

        The SDK's txRxPacket() transmits the write BEFORE reading the status
        packet, so a bad ack (COMM_RX_TIMEOUT/CORRUPT) does NOT prove the write
        missed the servo — the register may still have been applied. Per-servo
        state (attempted/confirmed/unknown/not_applied) is recorded so the
        caller knows exactly which physical servos may already be holding.
        """
        self._assert_connected()
        self.torque_write_attempted = []
        self.torque_write_confirmed = []
        self.torque_write_unknown = []
        self.torque_write_not_applied = []
        for sid in (ids or self.servo_ids):
            self.torque_write_attempted.append(sid)
            result, _err = self.packet_handler.write1ByteTxRx(
                sid, self.SMS_STS_TORQUE_ENABLE, value)
            if result == self._comm_success:
                self.torque_write_confirmed.append(sid)
                continue
            if result in (self._comm_rx_corrupt, self._comm_rx_timeout):
                self.torque_write_unknown.append(sid)
            else:
                self.torque_write_not_applied.append(sid)
            txrx = getattr(self.packet_handler, "getTxRxResult", None)
            text = f" ({txrx(result)})" if txrx else ""
            raise IOError(
                f"{verb} id={sid} comm result={result}{text} "
                f"attempted={self.torque_write_attempted} "
                f"confirmed={self.torque_write_confirmed} "
                f"unknown={self.torque_write_unknown} "
                f"not_applied={self.torque_write_not_applied}")

    def enable_torque(self, ids=None) -> None:
        """Leader motors hold position (torque enable register = 1)."""
        self._torque_write(1, "enable_torque", ids)

    def disable_torque(self, ids=None) -> None:
        """Leader motors free for hand motion (torque enable register = 0)."""
        self._torque_write(0, "disable_torque", ids)

    def read_torque_register(self, sid: int) -> tuple[int, int, int]:
        """READ-ONLY single read of the torque enable register (addr 40).

        Returns (value, comm_result, servo_error) like the SDK's read1ByteTxRx:
        value = 1 if that servo reports torque ENABLED, 0 if DISABLED, and 0 on
        comm failure (distinguish via comm_result). Never writes anything.
        """
        self._assert_connected()
        value, result, error = self.packet_handler.read1ByteTxRx(
            sid, self.SMS_STS_TORQUE_ENABLE)
        return value, result, error

    def read_servo_register(self, sid: int, addr: int, length: int = 1) -> tuple[int, int, int]:
        """READ-ONLY single-servo register read (never writes). length 1 uses
        read1ByteTxRx, length 2 read2ByteTxRx. Returns (value, comm_result,
        servo_error); value=0 on comm failure (distinguish via comm_result)."""
        self._assert_connected()
        if length == 1:
            return self.packet_handler.read1ByteTxRx(sid, addr)
        if length == 2:
            return self.packet_handler.read2ByteTxRx(sid, addr)
        raise ValueError(f"unsupported register length {length}")

    def read_voltage_registers(self, sid: int) -> dict:
        """READ-ONLY present/min/max voltage registers for one servo.
        Registers are raw bytes; Feetech unit is 0.1 V per LSB. Never writes."""
        present, p_res, p_err = self.read_servo_register(
            sid, self.SMS_STS_PRESENT_VOLTAGE, 1)
        mn, mn_res, mn_err = self.read_servo_register(
            sid, self.SMS_STS_MIN_VOLTAGE_LIMIT, 1)
        mx, mx_res, mx_err = self.read_servo_register(
            sid, self.SMS_STS_MAX_VOLTAGE_LIMIT, 1)
        return {
            "present_raw": present, "present_result": p_res, "present_error": p_err,
            "min_raw": mn, "min_result": mn_res, "min_error": mn_err,
            "max_raw": mx, "max_result": mx_res, "max_error": mx_err,
        }

    def read_servo_model(self, sid: int) -> tuple[int, int, int]:
        """READ-ONLY 2-byte model number (MODEL_L@3 / MODEL_H@4, little-endian)."""
        return self.read_servo_register(sid, self.SMS_STS_MODEL_L, 2)

    def write_goal(self, ticks_by_id: dict[int, int], speed: int = 1500, acc: int = 50) -> None:
        """Velocity/acc-limited goal position write (SMS_STS WritePosEx)."""
        self._assert_connected()
        for sid, ticks in ticks_by_id.items():
            if not (0 <= int(ticks) < 4096):
                raise ValueError(f"goal ticks {ticks} out of [0,4096) for id={sid}")
            result, _err = self.packet_handler.WritePosEx(int(sid), int(ticks), speed, acc)
            if result != self._comm_success:
                raise IOError(f"write_goal id={sid} ticks={ticks} comm result={result}")
