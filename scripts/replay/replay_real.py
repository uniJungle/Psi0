#!/usr/bin/env python3
"""Replay a LeRobot dataset episode on the real robot via C++ WBC ZMQ interface.

Usage:
    # 1. Start the robot-side C++ WBC controller (with zmq_manager input):
    bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

    # 2. (Optional) Stand up / handoff 5556:
    python scripts/replay/enable_control.py

    # 3. Replay body tokens via ZMQ; hands via DDS with --eef brainco|dex1
    #    (Do NOT run pico at the same time).
    python scripts/replay/replay_real.py --mode token --episode_idx 0 \
        --data_dir /path/to/halt_stand --eef brainco --dds-interface enp4s0
    python scripts/replay/replay_real.py --mode token --episode_idx 0 \
        --data_dir /path/to/test_dex1 --eef dex1 --dds-interface enp5s0

Architecture (same host as C++ --zmq-host, usually the workstation):
    [LeRobot Dataset] --> [replay_real.py] --bind--> tcp://*:5556 (PUB)
                                                         ^
                                                         | connect (SUB)
                                              [C++ WBC --zmq-host localhost]
                                                         |
                                              DDS / Unitree --> real robot body
    [teleop.*_hand_joints 2D] --> Brainco DDS --> brainco_hand.service --> fingers
    [teleop.*_hand_joints 1D] --> Dex1 DDS (rt/dex1/*/cmd) --> grippers

ZMQ Protocol:
    - planner mode: "planner" topic with upper_body_position (arm joints 14D)
    - token mode:   "pose" topic with token_state (64D motion token) + hand joints (14D)
    - command:      "command" topic (start/stop/planner mode)

Input Types (must match C++ --input-type):
    - zmq_manager: collect_psi0-sonic-data-manual.sh deploy  (receives command+pose)
    - manager:     deploy_psi0-sonic-rtc-robot.sh InterfaceManager
                   (starts in KEYBOARD; press Shift+3 for ZMQ, may need ENTER to
                    enable ZMQ stream — prefer zmq_manager for token replay)
"""

from __future__ import annotations

import json
import os
import sys
import time
import signal
import argparse
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add third_party/GR00T-WholeBodyControl to path for imports
_THIRD_PARTY = Path(__file__).parent.parent.parent / "third_party" / "GR00T-WholeBodyControl"
sys.path.insert(0, str(_THIRD_PARTY))
_UNITREE_SDK = _THIRD_PARTY / "external_dependencies" / "unitree_sdk2_python"
if _UNITREE_SDK.is_dir():
    sys.path.insert(0, str(_UNITREE_SDK))


def _ensure_dds_hand_deps() -> None:
    """Make unitree_sdk2py + cyclonedds (+ logging_mp for Dex1) importable in .venv-psi.

    Brainco/Dex1 DDS need unitree_sdk2py (vendored) and cyclonedds (usually installed
    in GR00T ``.venv_teleop``). Prefer borrowing that site-packages when missing.
    """
    if _UNITREE_SDK.is_dir() and str(_UNITREE_SDK) not in sys.path:
        sys.path.insert(0, str(_UNITREE_SDK))
    try:
        import cyclonedds  # noqa: F401
        import unitree_sdk2py  # noqa: F401
        return
    except ImportError:
        pass

    teleop_sp = _THIRD_PARTY / ".venv_teleop" / "lib" / "python3.10" / "site-packages"
    if teleop_sp.is_dir() and str(teleop_sp) not in sys.path:
        sys.path.insert(0, str(teleop_sp))
    try:
        import cyclonedds  # noqa: F401
        import unitree_sdk2py  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Brainco/Dex1 need unitree_sdk2py + cyclonedds. Install teleop deps:\n"
            "  cd third_party/GR00T-WholeBodyControl && bash install_scripts/install_pico.sh\n"
            "Or: uv pip install -e external_dependencies/unitree_sdk2_python && "
            "uv pip install cyclonedds\n"
            f"Original error: {exc}"
        ) from exc


# Backward-compatible alias
_ensure_brainco_deps = _ensure_dds_hand_deps


from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)
LOCOMOTION_MODE_IDLE = 0


FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    """Quantize motion token using FSQ (Finite Scalar Quantization)."""
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized


# ---------------- Action Extraction ----------------


def _as_1d(value: Any, fallback: np.ndarray) -> np.ndarray:
    if value is None:
        return fallback.copy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _pad_hand_joints_to_dex3(hand: np.ndarray) -> np.ndarray:
    """Pad hand joints to Dex3 7D wire format expected by ZMQ Protocol v4.

    Brainco datasets store 2D ``[thumb_aux, others]``; C++ only accepts shape
    ``[7]`` / ``[N, 7]``. Pad with zeros so body tokens still replay.
    """
    hand = np.asarray(hand, dtype=np.float32).reshape(-1)
    if hand.size >= 7:
        return hand[:7].astype(np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[: hand.size] = hand
    return out


def extract_brainco_2d(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract Brainco 2D targets ``[thumb_aux, others]`` from a frame."""
    left = _as_1d(frame.get("teleop.left_hand_joints"), np.zeros(2))
    right = _as_1d(frame.get("teleop.right_hand_joints"), np.zeros(2))
    # Psi0 68D action: motion_token(64) + left(2) + right(2)
    if left.size < 2 or right.size < 2:
        action68 = _as_1d(frame.get("action"), np.zeros(0))
        if action68.size >= 68:
            left = action68[64:66]
            right = action68[66:68]
    # Fall back to tail of observation.state / action.wbc / states (33D brainco layout).
    if left.size < 2:
        state = _as_1d(
            frame.get("action.wbc"),
            _as_1d(frame.get("states"), _as_1d(frame.get("observation.state"), np.zeros(33))),
        )
        if state.size >= 33:
            left = state[29:31]
            right = state[31:33]
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2:
        left = np.zeros(2, dtype=np.float64)
    if right.size < 2:
        right = np.zeros(2, dtype=np.float64)
    return left[:2], right[:2]


def extract_dex1_1d(frame: dict[str, Any]) -> tuple[float, float]:
    """Extract Dex1 1D gripper commands from a frame (already scaled, typically [0, 5.5])."""
    left = _as_1d(frame.get("teleop.left_hand_joints"), np.zeros(0))
    right = _as_1d(frame.get("teleop.right_hand_joints"), np.zeros(0))
    # Fall back to tail of observation.state / action.wbc (31D = body29 + L1 + R1).
    if left.size < 1 or right.size < 1:
        state = _as_1d(
            frame.get("action.wbc"),
            _as_1d(frame.get("states"), _as_1d(frame.get("observation.state"), np.zeros(31))),
        )
        if state.size >= 31:
            if left.size < 1:
                left = state[29:30]
            if right.size < 1:
                right = state[30:31]
    left_v = float(left.reshape(-1)[0]) if left.size >= 1 else 0.0
    right_v = float(right.reshape(-1)[0]) if right.size >= 1 else 0.0
    return left_v, right_v


def extract_action_token(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract motion token and hand joints from a dataset frame.

    Supports:
      - raw SONIC: action.motion_token + teleop.*_hand_joints
      - Psi0 lerobot_v2.1: action[68] = token(64) + hand(4)
    """
    motion_token = _as_1d(frame.get("action.motion_token"), np.zeros(0))
    left_hand = _as_1d(frame.get("teleop.left_hand_joints"), np.zeros(0))
    right_hand = _as_1d(frame.get("teleop.right_hand_joints"), np.zeros(0))

    action68 = _as_1d(frame.get("action"), np.zeros(0))
    if motion_token.size < 64 and action68.size >= 64:
        motion_token = action68[:64]
    if left_hand.size < 2 and action68.size >= 66:
        left_hand = action68[64:66]
    if right_hand.size < 2 and action68.size >= 68:
        right_hand = action68[66:68]

    if motion_token.size < 64:
        motion_token = np.zeros(64, dtype=np.float64)
    left_hand = _pad_hand_joints_to_dex3(left_hand if left_hand.size else np.zeros(7))
    right_hand = _pad_hand_joints_to_dex3(right_hand if right_hand.size else np.zeros(7))
    return motion_token[:64], left_hand, right_hand


def create_brainco_hand(dds_interface: str):
    """Create active Brainco DDS driver for hand replay (same path as pico teleop)."""
    _ensure_dds_hand_deps()
    from eef.brainco.brainco import Brainco

    print(
        f"[ReplayReal] Initializing Brainco "
        f"(dds_interface={dds_interface or 'default'})..."
    )
    hand = Brainco(passive=False, network_interface=dds_interface or None)
    hand.change_open_pose(801)
    hand.set_gripper_targets(0.0, 0.0)
    print("[ReplayReal] Brainco ready — will replay teleop.*_hand_joints via DDS")
    return hand


def shutdown_brainco_hand(hand) -> None:
    if hand is None:
        return
    try:
        hand.set_gripper_targets(0.0, 0.0)
        time.sleep(0.1)
    except Exception as exc:
        print(f"[ReplayReal] Brainco open-on-exit failed: {exc}")
    try:
        hand.close()
    except Exception as exc:
        print(f"[ReplayReal] Brainco close failed: {exc}")
    print("[ReplayReal] Brainco shutdown complete")


def create_dex1_hand(dds_interface: str):
    """Create active Dex1 DDS driver for gripper replay (same path as pico teleop)."""
    _ensure_dds_hand_deps()
    from eef.dex1.dex1 import Dex1

    print(
        f"[ReplayReal] Initializing Dex1 "
        f"(dds_interface={dds_interface or 'default'})..."
    )
    hand = Dex1(network_interface=dds_interface or None)
    hand.set_gripper_ratios(0.0, 0.0)
    print("[ReplayReal] Dex1 ready — will replay teleop.*_hand_joints (1D) via DDS")
    return hand


def shutdown_dex1_hand(hand) -> None:
    if hand is None:
        return
    try:
        hand.set_gripper_ratios(0.0, 0.0)
        time.sleep(0.1)
    except Exception as exc:
        print(f"[ReplayReal] Dex1 reset-on-exit failed: {exc}")
    print("[ReplayReal] Dex1 shutdown complete")


def extract_action_joints(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract joint values from a dataset frame (for planner mode)."""
    # observation.state contains all joint positions (43D)
    state = _as_1d(frame.get("observation.state"), np.zeros(43))

    # Split into body parts
    action = {
        "left_leg": state[0:6],
        "right_leg": state[6:12],
        "waist": state[12:15],
        "left_arm": state[15:22],
        "left_hand": state[22:29],
        "right_arm": state[29:36],
        "right_hand": state[36:43],
    }
    return action


def action_to_planner_fields(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert action dict to planner message fields."""
    # Upper body = waist + arms + hands
    upper_body = np.concatenate([
        action["waist"],       # 3
        action["left_arm"],    # 7
        action["left_hand"],  # 7
        action["right_arm"],   # 7
        action["right_hand"],  # 7
    ])  # total: 31

    # Left hand: arm (7) + hand (7) = 14
    left_hand_position = np.concatenate([action["left_arm"], action["left_hand"]])

    # Right hand: arm (7) + hand (7) = 14
    right_hand_position = np.concatenate([action["right_arm"], action["right_hand"]])

    return {
        "upper_body_position": upper_body,
        "left_hand_position": left_hand_position,
        "right_hand_position": right_hand_position,
    }


def _load_dataset_info(data_dir: str | Path) -> dict[str, Any]:
    info_path = Path(data_dir) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"info.json not found: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_episode_parquet(data_dir: str | Path, episode_idx: int) -> Path:
    """Resolve parquet path for episode_idx from local LeRobot meta/info.json."""
    data_path = Path(data_dir)
    info = _load_dataset_info(data_path)

    total = int(info.get("total_episodes", 0))
    if episode_idx < 0 or (total > 0 and episode_idx >= total):
        raise ValueError(f"Episode index {episode_idx} out of range, available: 0-{max(total - 1, 0)}")

    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = episode_idx // chunks_size
    data_tpl = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    parquet_path = data_path / data_tpl.format(
        episode_chunk=chunk_idx,
        episode_index=episode_idx,
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    return parquet_path


def resolve_episode_task_prompt(data_dir: str | Path, episode_idx: int) -> str:
    """Load task prompt text for episode_idx from meta/episodes.jsonl (fallback: tasks.jsonl)."""
    data_path = Path(data_dir)
    episodes_path = data_path / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                if int(ep.get("episode_index", -1)) != episode_idx:
                    continue
                tasks = ep.get("tasks") or []
                if isinstance(tasks, list) and tasks:
                    return str(tasks[0])
                if isinstance(tasks, str) and tasks:
                    return tasks
                break

    tasks_path = data_path / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                task = item.get("task")
                if task:
                    return str(task)
                break
    return ""


def resolve_episode_videos(
    data_dir: str | Path,
    episode_idx: int,
) -> list[tuple[str, Path]]:
    """Resolve mp4 paths for an episode (LeRobot video_path template)."""
    data_path = Path(data_dir)
    info = _load_dataset_info(data_path)

    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = episode_idx // chunks_size
    video_tpl = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )

    features = info.get("features", {})
    preferred = [
        "observation.images.ego_view_left",
        "observation.images.ego_view",
        "observation.images.ego_view_right",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    found = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    video_keys = [k for k in preferred if k in found]
    # Keep any remaining video keys after preferred order.
    for key in sorted(found):
        if key not in video_keys:
            video_keys.append(key)

    resolved: list[tuple[str, Path]] = []
    for key in video_keys:
        path = data_path / video_tpl.format(
            episode_chunk=chunk_idx,
            episode_index=episode_idx,
            video_key=key,
        )
        if path.is_file():
            resolved.append((key, path))
        else:
            print(f"[ReplayReal] Video not found, skip: {path}")
    return resolved


class EpisodeVideoPreview:
    """OpenCV window synced to replay frame index (one step == one video frame)."""

    WINDOW_NAME = "Replay Video"

    def __init__(
        self,
        videos: list[tuple[str, Path]],
        task_prompt: str = "",
        episode_idx: int = 0,
    ):
        import cv2

        self._cv2 = cv2
        self.task_prompt = (task_prompt or "").strip()
        self.episode_idx = episode_idx
        self.caps: list[tuple[str, Any]] = []
        for key, path in videos:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"[ReplayReal] Failed to open video: {path}")
                continue
            self.caps.append((key, cap))
            print(f"[ReplayReal] Video preview: {key} -> {path}")

        if not self.caps:
            raise RuntimeError("No playable episode videos found")

        if self.task_prompt:
            print(f"[ReplayReal] Task prompt: {self.task_prompt}")

        n_views = len(self.caps)
        # Always one row: 2-view → 1280x480; 4-view → wider single strip.
        win_w = 1280 if n_views <= 2 else 1920
        win_h = 480
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, win_w, win_h)

    def _draw_overlay(self, canvas: Any, frame_idx: int) -> None:
        """Draw frame index + task-prompt on the concatenated preview canvas."""
        self._cv2.putText(
            canvas,
            f"ep{self.episode_idx}  #{frame_idx}",
            (12, 28),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            self._cv2.LINE_AA,
        )
        if not self.task_prompt:
            return

        # Wrap long prompts to fit canvas width.
        max_chars = max(24, canvas.shape[1] // 11)
        text = self.task_prompt
        lines = []
        while text:
            lines.append(text[:max_chars])
            text = text[max_chars:]
            if len(lines) >= 3:
                if text:
                    lines[-1] = lines[-1][:-3] + "..."
                break

        y0 = 58
        for i, line in enumerate(lines):
            y = y0 + i * 28
            # Dark outline for readability on bright frames.
            self._cv2.putText(
                canvas,
                line,
                (12, y),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                self._cv2.LINE_AA,
            )
            self._cv2.putText(
                canvas,
                line,
                (12, y),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                self._cv2.LINE_AA,
            )

    @staticmethod
    def _hstack_match_height(cv2, frames: list[Any]) -> Any:
        h = min(f.shape[0] for f in frames)
        resized = []
        for f in frames:
            if f.shape[0] != h:
                scale = h / f.shape[0]
                w = max(1, int(f.shape[1] * scale))
                f = cv2.resize(f, (w, h))
            resized.append(f)
        return np.concatenate(resized, axis=1) if len(resized) > 1 else resized[0]

    def show_frame(self, frame_idx: int) -> None:
        frames = []
        for key, cap in self.caps:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            label = key.rsplit(".", 1)[-1]
            self._cv2.putText(
                frame,
                label,
                (12, frame.shape[0] - 16),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                self._cv2.LINE_AA,
            )
            frames.append(frame)

        if not frames:
            return

        # One row, N columns (ego L|R then wrists L|R when all four present).
        canvas = self._hstack_match_height(self._cv2, frames)
        self._draw_overlay(canvas, frame_idx)
        self._cv2.imshow(self.WINDOW_NAME, canvas)
        self._cv2.waitKey(1)

    def close(self) -> None:
        for _, cap in self.caps:
            cap.release()
        self.caps.clear()
        try:
            self._cv2.destroyWindow(self.WINDOW_NAME)
            self._cv2.waitKey(1)
        except Exception:
            pass


def row_to_frame(row: pd.Series) -> dict[str, Any]:
    """Convert a parquet row to a plain dict of numpy arrays."""
    frame: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            frame[key] = np.asarray(value)
        else:
            frame[key] = value
    return frame


# ---------------- ZMQ Client ----------------


class ReplayZMQClient:
    """ZMQ client for sending replay commands to C++ WBC."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5559,
        mode: str = "planner",
        input_type: str = "zmq_manager",
        verbose: bool = True,
    ):
        """
        Args:
            host: Robot/workstation IP (localhost for simulation)
            port: ZMQ port
            mode: "planner" or "token"
            input_type: "zmq_manager" (auto-start) or "manager" (manual start)
            verbose: Print debug info
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.input_type = input_type
        self.verbose = verbose

        self.ctx = None
        self.sock = None
        self._frame_index = 0

    def connect(self):
        """Bind ZMQ PUB so C++ zmq_manager / InterfaceManager can SUB-connect.

        SONIC C++ side does ``socket.connect(tcp://{zmq_host}:{port})`` (default
        ``localhost:5556``). Publishers (pico_manager, psi_rtc_sonic_client,
        test_zmq_manager) therefore must ``bind``. Connecting as PUB to the
        robot IP is wrong and messages are silently dropped.
        """
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        # host="*" or empty -> bind all interfaces; else bind specific address
        bind_host = self.host if self.host not in ("localhost", "127.0.0.1") else "*"
        if bind_host in ("", "auto"):
            bind_host = "*"
        endpoint = f"tcp://{bind_host}:{self.port}"
        self.sock.bind(endpoint)

        # Give C++ SUB time to (re)connect after bind
        time.sleep(1.0)
        if self.verbose:
            print(f"[ReplayZMQ] Bound PUB at {endpoint}, mode={self.mode}, input_type={self.input_type}")
            print("[ReplayZMQ] Ensure C++ deploy --zmq-host points at this machine (default: localhost)")

    def send_command(self, start: bool = False, stop: bool = False, planner: bool = True):
        """Send command message (same wire format as pico_manager / replay_sim)."""
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self.sock.send(msg)
        if self.verbose:
            print(f"[ReplayZMQ] Command: start={start}, stop={stop}, planner={planner}")

    def send_action(self, action: dict[str, float]):
        """Send action frame via planner topic (direct joint values)."""
        fields = action_to_planner_fields(action)
        msg = build_planner_message(
            mode=0,
            movement=[0.0, 0.0, 0.0],
            facing=[0.0, 0.0, 0.0],
            upper_body_position=fields["upper_body_position"],
            left_hand_position=fields["left_hand_position"],
            right_hand_position=fields["right_hand_position"],
        )
        self.sock.send(msg)

    def send_token(self, motion_token: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray):
        """
        Send motion token via pose topic (Protocol v4).

        Args:
            motion_token: (64D) quantized motion token
            left_hand: (7D) left hand joints
            right_hand: (7D) right hand joints
        """
        if motion_token.ndim > 1:
            motion_token = motion_token[0]
        left_hand = _pad_hand_joints_to_dex3(left_hand)
        right_hand = _pad_hand_joints_to_dex3(right_hand)

        # FSQ quantize the motion token
        token_qtz = fsq_quantize(np.asarray(motion_token, dtype=np.float32).reshape(-1))
        if token_qtz.size != 64:
            raise ValueError(f"motion_token must be 64D, got shape {token_qtz.shape}")

        # Build pose message: token(64) + left hand(7) + right hand(7)
        pose_data = {
            "token_state": token_qtz.reshape(1, 64).astype(np.float32),
            "left_hand_joints": left_hand.reshape(1, 7).astype(np.float32),
            "right_hand_joints": right_hand.reshape(1, 7).astype(np.float32),
        }
        pose_msg = pack_pose_message(pose_data, topic="pose", version=4)
        self.sock.send(pose_msg)

        self._frame_index += 1

    def send_idle_planner(self):
        """Send idle planner command (robot stands still in PLANNER mode)."""
        msg = build_planner_message(
            mode=LOCOMOTION_MODE_IDLE,
            movement=[0.0, 0.0, 0.0],
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0,
        )
        self.sock.send(msg)

    def release(self, send_stop: bool = False):
        """Close PUB socket. Default: do not send stop (deploy keeps running)."""
        if self.sock:
            if send_stop:
                self.send_command(start=False, stop=True, planner=True)
                time.sleep(0.1)
            elif self.verbose:
                print(
                    "[ReplayZMQ] Releasing port without stop "
                    "(deploy stays in PLANNER/CONTROL)"
                )
            self.sock.close(linger=0)
            self.sock = None
        if self.ctx:
            self.ctx.term()
            self.ctx = None
        if self.verbose:
            print("[ReplayZMQ] Stopped")

    def stop(self):
        """Legacy alias — release without stopping deploy."""
        self.release(send_stop=False)


# ---------------- Main Replay ----------------


class ReplayReal:
    """Replay a dataset episode on the real robot."""

    def __init__(
        self,
        data_dir: str,
        episode_idx: int = 0,
        fps: int = 30,
        robot_ip: str = "192.168.123.164",
        zmq_port: int = 5559,
        mode: str = "token",
        input_type: str = "zmq_manager",
        warmup_seconds: float = 2.0,
        handoff_seconds: float = 2.0,
        stop_on_exit: bool = False,
        eef: str = "none",
        dds_interface: str = "enp4s0",
        show_video: bool = True,
        no_robot: bool = False,
    ):
        """
        Args:
            data_dir: Path to LeRobot dataset directory
            episode_idx: Episode index to replay
            fps: Target replay FPS
            robot_ip: Robot's IP address for ZMQ connection
            zmq_port: ZMQ port on robot/workstation
            mode: "planner" for direct joint values, "token" for motion_token
            input_type: "zmq_manager" (auto-start on first pose) or "manager" (manual)
            warmup_seconds: Time to wait after start command
            eef: "none", "brainco", or "dex1"
            dds_interface: NIC for Brainco/Dex1 DDS
            show_video: Open OpenCV window with episode ego videos
            no_robot: Preview-only mode; skip ZMQ / DDS and ignore eef
        """
        self.data_dir = data_dir
        self.episode_idx = episode_idx
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.mode = mode
        self.input_type = input_type
        self.warmup_seconds = warmup_seconds
        self.handoff_seconds = handoff_seconds
        self.stop_on_exit = stop_on_exit
        self.eef = eef
        self.dds_interface = dds_interface
        self.show_video = show_video
        self.no_robot = no_robot
        self.running = True
        self._handoff_done = False
        self.brainco_hand = None
        self.dex1_hand = None
        self.video_preview: Optional[EpisodeVideoPreview] = None
        self.zmq: Optional[ReplayZMQClient] = None

        # Read selected episode parquet directly (no LeRobotDataset / video preload)
        self.parquet_path = resolve_episode_parquet(data_dir, episode_idx)
        try:
            self.df = pd.read_parquet(self.parquet_path)
        except OSError as exc:
            import pyarrow

            raise RuntimeError(
                f"Failed to read parquet with pyarrow {pyarrow.__version__} "
                f"({sys.executable}): {exc}\n"
                "Use the project venv (pyarrow>=21), e.g.:\n"
                "  cd ~/ycb_ws/Psi0 && source .venv-psi/bin/activate\n"
                "Anaconda base pyarrow 19 often fails with "
                "'Repetition level histogram size mismatch'."
            ) from exc
        self.num_frames = len(self.df)
        if self.num_frames == 0:
            raise ValueError(f"Empty episode parquet: {self.parquet_path}")
        print(
            f"[ReplayReal] Loaded parquet: {self.parquet_path} "
            f"(episode={episode_idx}, frames={self.num_frames})"
        )

        if self.no_robot:
            if self.eef != "none":
                print(f"[ReplayReal] --no-robot enabled; ignoring --eef={self.eef}")
            self.eef = "none"
            print("[ReplayReal] Preview-only mode: skipping robot ZMQ and hand DDS")
        else:
            if self.eef == "brainco":
                self.brainco_hand = create_brainco_hand(self.dds_interface)
            elif self.eef == "dex1":
                self.dex1_hand = create_dex1_hand(self.dds_interface)
            elif self.eef not in ("", "none"):
                raise ValueError(
                    f"Unsupported --eef={self.eef!r}. Use 'none', 'brainco', or 'dex1'."
                )

            # ZMQ PUB must bind on the same host that C++ --zmq-host connects to.
            # Default deploy uses --zmq-host localhost, so bind *:5556 on this machine.
            # (robot_ip is kept for API compatibility; ZMQ does not target the robot NIC.)
            self.zmq = ReplayZMQClient(
                host="*",
                port=zmq_port,
                mode=mode,
                input_type=input_type,
            )
            self.zmq.connect()

        if self.show_video:
            videos = resolve_episode_videos(data_dir, episode_idx)
            task_prompt = resolve_episode_task_prompt(data_dir, episode_idx)
            if videos:
                try:
                    self.video_preview = EpisodeVideoPreview(
                        videos,
                        task_prompt=task_prompt,
                        episode_idx=episode_idx,
                    )
                except Exception as exc:
                    print(f"[ReplayReal] Video preview disabled: {exc}")
            else:
                print("[ReplayReal] No episode videos found; continuing without preview")
                if task_prompt:
                    print(f"[ReplayReal] Task prompt: {task_prompt}")

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print(f"\n[ReplayReal] Signal {sig}, stopping replay loop...")
        self.running = False

    def _shutdown_hands(self) -> None:
        shutdown_brainco_hand(self.brainco_hand)
        self.brainco_hand = None
        shutdown_dex1_hand(self.dex1_hand)
        self.dex1_hand = None

    def _handoff_to_idle_planner(self):
        """After replay: PLANNER + idle, release port, keep deploy running."""
        if self._handoff_done:
            return
        self._handoff_done = True

        if self.zmq is None or self.zmq.sock is None:
            self._shutdown_hands()
            if self.video_preview is not None:
                self.video_preview.close()
                self.video_preview = None
            return

        if self.stop_on_exit:
            print("[ReplayReal] Sending stop command (--stop-on-exit)...")
            self.zmq.release(send_stop=True)
            self._shutdown_hands()
            if self.video_preview is not None:
                self.video_preview.close()
                self.video_preview = None
            return

        print(
            "[ReplayReal] Handoff: switch to PLANNER + idle "
            f"({self.handoff_seconds}s), then release port (deploy keeps running)..."
        )
        self.zmq.send_command(start=True, stop=False, planner=True)
        time.sleep(0.5)

        deadline = time.perf_counter() + max(0.0, self.handoff_seconds)
        interval = 1.0 / 30.0
        while time.perf_counter() < deadline:
            self.zmq.send_idle_planner()
            time.sleep(interval)

        self.zmq.release(send_stop=False)
        print("[ReplayReal] Done — deploy still running in idle PLANNER")
        self._shutdown_hands()
        if self.video_preview is not None:
            self.video_preview.close()
            self.video_preview = None

    def run(self):
        """Run the replay."""
        print(f"[ReplayReal] Starting replay at {self.fps} Hz, mode={self.mode}, input_type={self.input_type}")
        if self.brainco_hand is not None:
            print("[ReplayReal] Brainco hand replay enabled (DDS 2D targets)")
        if self.dex1_hand is not None:
            print("[ReplayReal] Dex1 gripper replay enabled (DDS 1D targets)")
        if self.video_preview is not None:
            print("[ReplayReal] Video preview window: 'Replay Video' (synced to frame index)")
        if self.no_robot:
            print("[ReplayReal] --no-robot: local preview only, no commands will be sent")

        if not self.no_robot:
            # ZMQManager only processes `start` while in PLANNER mode. Sending
            # start=True with planner=False immediately jumps to STREAMED_MOTION and
            # drops the start handshake → C++ stays in WAIT_FOR_CONTROL (tokens
            # arrive/log but motors never engage). Match pico_manager / RTC client:
            #   1) start + planner=True  → enter CONTROL
            #   2) start + planner=False → STREAMED_MOTION (token / pose topic)
            #   3) stream pose frames
            print("[ReplayReal] Step 1/2: start control in PLANNER mode...")
            self.zmq.send_command(start=True, stop=False, planner=True)
            time.sleep(max(self.warmup_seconds, 2.0))

            if self.mode == "token":
                print("[ReplayReal] Step 2/2: switch to STREAMED_MOTION (pose/token)...")
                self.zmq.send_command(start=True, stop=False, planner=False)
                time.sleep(1.0)

        try:
            frame_idx = 0
            prev_time = time.perf_counter()

            while self.running and frame_idx < self.num_frames:
                frame = row_to_frame(self.df.iloc[frame_idx])

                if not self.no_robot:
                    if self.mode == "token":
                        motion_token, left_hand, right_hand = extract_action_token(frame)
                        self.zmq.send_token(motion_token, left_hand, right_hand)
                    else:
                        action = extract_action_joints(frame)
                        self.zmq.send_action(action)

                if self.brainco_hand is not None:
                    left_2d, right_2d = extract_brainco_2d(frame)
                    self.brainco_hand.set_2d_targets(left_2d, right_2d)
                if self.dex1_hand is not None:
                    left_1d, right_1d = extract_dex1_1d(frame)
                    self.dex1_hand.set_gripper_ratios(left_1d, right_1d)

                if self.video_preview is not None:
                    self.video_preview.show_frame(frame_idx)

                if frame_idx % 30 == 0:
                    print(f"[ReplayReal] Frame {frame_idx}/{self.num_frames}")

                elapsed = time.perf_counter() - prev_time
                sleep_time = self.frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                prev_time = time.perf_counter()
                frame_idx += 1

            if frame_idx >= self.num_frames:
                print("[ReplayReal] Replay finished")
        finally:
            self._handoff_to_idle_planner()


def main():
    parser = argparse.ArgumentParser(
        description="Replay a LeRobot dataset episode on the real robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With collect_psi0-sonic-data-manual.sh deploy (input_type=zmq_manager):
  python replay_real.py --mode token --episode_idx 0 --robot_ip 192.168.123.164

  # With deploy_psi0-sonic-rtc-robot.sh (input_type=manager):
  python replay_real.py --mode token --episode_idx 0 --robot_ip 192.168.123.164 --input_type manager

  # Planner mode (direct joint values):
  python replay_real.py --mode planner --episode_idx 0 --robot_ip 192.168.123.164
"""
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/origin",
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Episode index to replay",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Replay frame rate (Hz). Dataset fps is typically 30.",
    )
    parser.add_argument(
        "--robot_ip",
        type=str,
        default="192.168.123.164",
        help="Unused for ZMQ (kept for compatibility). C++ must use --zmq-host localhost.",
    )
    parser.add_argument(
        "--zmq_port",
        type=int,
        default=5556,
        help="ZMQ port (default: 5556; must match deploy --zmq-port / enable_control)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="token",
        choices=["planner", "token"],
        help="Replay mode: 'planner' for joint values, 'token' for motion tokens",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="zmq_manager",
        choices=["zmq_manager", "manager"],
        help="""
Input type:
  zmq_manager - Used by collect_psi0-sonic-data-manual.sh deploy (auto-start on first pose)
  manager     - Used by deploy_psi0-sonic-rtc-robot.sh (manual start/stop)
""",
    )
    parser.add_argument(
        "--warmup_seconds",
        type=float,
        default=2.0,
        help="Warmup time after start command (for manager mode)",
    )
    parser.add_argument(
        "--handoff-seconds",
        type=float,
        default=2.0,
        help="After replay: idle PLANNER duration before releasing port (default: 2.0)",
    )
    parser.add_argument(
        "--stop-on-exit",
        action="store_true",
        help="Send stop=True on exit (shuts down deploy; default: handoff to idle PLANNER)",
    )
    parser.add_argument(
        "--eef",
        type=str,
        default="none",
        choices=["none", "brainco", "dex1"],
        help="End-effector DDS replay: none, brainco (2D), or dex1 (1D)",
    )
    parser.add_argument(
        "--dds-interface",
        type=str,
        default="enp4s0",
        help="NIC for Brainco/Dex1 DDS (default: enp4s0)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable OpenCV episode video preview window",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Preview only: skip ZMQ / DDS and ignore --eef",
    )

    args = parser.parse_args()

    # Create and run replay
    replay = ReplayReal(
        data_dir=args.data_dir,
        episode_idx=args.episode_idx,
        fps=args.fps,
        robot_ip=args.robot_ip,
        zmq_port=args.zmq_port,
        mode=args.mode,
        input_type=args.input_type,
        warmup_seconds=args.warmup_seconds,
        handoff_seconds=args.handoff_seconds,
        stop_on_exit=args.stop_on_exit,
        eef=args.eef,
        dds_interface=args.dds_interface,
        show_video=not args.no_video,
        no_robot=args.no_robot,
    )
    try:
        replay.run()
    except KeyboardInterrupt:
        print("\n[ReplayReal] Interrupted")
        replay._handoff_to_idle_planner()
    finally:
        shutdown_brainco_hand(getattr(replay, "brainco_hand", None))
        shutdown_dex1_hand(getattr(replay, "dex1_hand", None))

if __name__ == "__main__":
    # Import zmq and json at top level
    import zmq
    import json
    import struct
    main()
