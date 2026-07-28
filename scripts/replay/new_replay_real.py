#!/usr/bin/env python3
"""Interactive Replay for real robot via C++ WBC ZMQ interface.

Features:
    - 1: Toggle between IDLE_PLANER and SLOW_WALK modes
    - W/A/S/D: Move (in SLOW_WALK mode, relative to facing direction)
    - Q/E: Turn left/right
    - Z/C: Navigate to previous/next episode
    - Enter: Play/Stop current episode
    - Ctrl+C: Exit (returns to idle planner)

Usage:
    # 1. Start the robot-side C++ WBC controller:
    bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

    # 2. Run interactive replay:
    python scripts/replay/new_replay_real.py \
        --data_dir /path/to/dataset \
        --episode_idx 0 \
        --zmq_port 5556 \
        --mode token \
        --eef none
"""

from __future__ import annotations

import json
import math
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
from enum import Enum
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

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# Constants for locomotion modes
LOCOMOTION_MODE_IDLE = 0
LOCOMOTION_MODE_SLOW_WALK = 2


# ---------------- FSQ Quantization (for motion token) ----------------
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    """Quantize motion token using FSQ (Finite Scalar Quantization)."""
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized


# ---------------- Keyboard Controller ----------------


class NonBlockingKeyboard:
    """Terminal keyboard input listener (non-blocking, independent thread)."""

    def __init__(self):
        self._running = True
        self._key_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._original_termios: Optional[tuple] = None

    def _setup_terminal(self):
        """Set terminal to non-canonical, non-blocking mode."""
        self._original_termios = termios.tcgetattr(sys.stdin)
        new_attrs = termios.tcgetattr(sys.stdin)
        new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON | termios.ISIG)
        new_attrs[6][termios.VMIN] = 0
        new_attrs[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin, termios.TCSANOW, new_attrs)

    def _restore_terminal(self):
        """Restore original terminal settings."""
        if self._original_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, self._original_termios)

    def _read_loop(self):
        """Background thread that reads keyboard input."""
        while self._running:
            try:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._key_queue.put(ch)
            except Exception:
                break
            time.sleep(0.01)

    def start(self):
        """Start the keyboard listener thread."""
        self._setup_terminal()
        self._drain_input()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[Keyboard] Control enabled")

    def _drain_input(self):
        """Drain any buffered input from stdin."""
        while select.select([sys.stdin], [], [], 0.01)[0]:
            sys.stdin.read(10)

    def stop(self):
        """Stop the keyboard listener and restore terminal."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        self._restore_terminal()
        print("[Keyboard] Control disabled")

    def get_key(self) -> Optional[str]:
        """Get the next key press (non-blocking). Returns None if no key pressed."""
        try:
            return self._key_queue.get_nowait()
        except queue.Empty:
            return None


# ---------------- Interactive Replay Mode ----------------


class InteractiveReplayMode(Enum):
    IDLE_PLANER = "idle_planner"
    SLOW_WALK = "slow_walk"
    PLAYING = "playing"


# ---------------- Action Extraction ----------------


def _as_1d(value: Any, fallback: np.ndarray) -> np.ndarray:
    if value is None:
        return fallback.copy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _pad_hand_joints_to_dex3(hand: np.ndarray) -> np.ndarray:
    """Pad hand joints to Dex3 7D wire format expected by ZMQ Protocol v4."""
    hand = np.asarray(hand, dtype=np.float32).reshape(-1)
    if hand.size >= 7:
        return hand[:7].astype(np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[: hand.size] = hand
    return out


def extract_action_token(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract motion token and hand joints from a dataset frame."""
    motion_token = _as_1d(frame.get("action.motion_token"), np.zeros(64))
    left_hand = _pad_hand_joints_to_dex3(
        _as_1d(frame.get("teleop.left_hand_joints"), np.zeros(7))
    )
    right_hand = _pad_hand_joints_to_dex3(
        _as_1d(frame.get("teleop.right_hand_joints"), np.zeros(7))
    )
    return motion_token, left_hand, right_hand


def extract_action_joints(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract joint values from a dataset frame (for planner mode)."""
    state = _as_1d(frame.get("observation.state"), np.zeros(43))
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
    upper_body = np.concatenate([
        action["waist"],
        action["left_arm"],
        action["left_hand"],
        action["right_arm"],
        action["right_hand"],
    ])
    left_hand_position = np.concatenate([action["left_arm"], action["left_hand"]])
    right_hand_position = np.concatenate([action["right_arm"], action["right_hand"]])
    return {
        "upper_body_position": upper_body,
        "left_hand_position": left_hand_position,
        "right_hand_position": right_hand_position,
    }


# ---------------- Dataset Utilities ----------------


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


def row_to_frame(row) -> dict[str, Any]:
    """Convert a pandas row to a dict with numpy arrays."""
    frame = {}
    for col in row.index:
        val = row[col]
        if hasattr(val, 'to_numpy'):
            frame[col] = val.to_numpy()
        elif hasattr(val, 'numpy'):
            frame[col] = val.numpy()
        else:
            frame[col] = np.asarray(val) if val is not None else None
    return frame


# ---------------- ZMQ Client ----------------


class ReplayZMQClient:
    """ZMQ PUB client for sending commands/poses to C++ WBC."""

    def __init__(self, host: str = "*", port: int = 5556, mode: str = "token", input_type: str = "zmq_manager"):
        self.host = host
        self.port = port
        self.mode = mode
        self.input_type = input_type
        self.ctx = None
        self.sock = None
        self.verbose = True
        self._frame_index = 0

    def connect(self):
        """Create and bind ZMQ PUB socket."""
        import zmq
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        addr = f"tcp://{self.host}:{self.port}"
        self.sock.bind(addr)
        print(f"[ReplayZMQ] Bound to {addr}")
        time.sleep(0.5)

    def send_command(self, start: bool = False, stop: bool = False, planner: bool = True):
        """Send command message."""
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
        """Send motion token via pose topic (Protocol v4)."""
        if motion_token.ndim > 1:
            motion_token = motion_token[0]
        left_hand = _pad_hand_joints_to_dex3(left_hand)
        right_hand = _pad_hand_joints_to_dex3(right_hand)

        token_qtz = fsq_quantize(np.asarray(motion_token, dtype=np.float32).reshape(-1))
        if token_qtz.size != 64:
            raise ValueError(f"motion_token must be 64D, got shape {token_qtz.shape}")

        pose_data = {
            "token_state": token_qtz.reshape(1, 64).astype(np.float32),
            "left_hand_joints": left_hand.reshape(1, 7).astype(np.float32),
            "right_hand_joints": right_hand.reshape(1, 7).astype(np.float32),
        }
        pose_msg = pack_pose_message(pose_data, topic="pose", version=4)
        self.sock.send(pose_msg)
        self._frame_index += 1

    def send_idle_planner(self, facing_x: float = 1.0, facing_y: float = 0.0):
        """Send idle planner command with facing direction."""
        msg = build_planner_message(
            mode=LOCOMOTION_MODE_IDLE,
            movement=[0.0, 0.0, 0.0],
            facing=[facing_x, facing_y, 0.0],
            speed=-1.0,
            height=-1.0,
        )
        self.sock.send(msg)

    def send_slow_walk(self, movement: list, facing_x: float, facing_y: float, speed: float):
        """Send slow walk command with movement and facing direction."""
        msg = build_planner_message(
            mode=LOCOMOTION_MODE_SLOW_WALK,
            movement=[movement[0], movement[1], 0.0],
            facing=[facing_x, facing_y, 0.0],
            speed=speed,
            height=-1.0,
        )
        self.sock.send(msg)

    def release(self, send_stop: bool = False):
        """Close PUB socket."""
        if self.sock:
            if send_stop:
                self.send_command(start=False, stop=True, planner=True)
                time.sleep(0.1)
            self.sock.close(linger=0)
            self.sock = None
        if self.ctx:
            self.ctx.term()
            self.ctx = None
        if self.verbose:
            print("[ReplayZMQ] Stopped")


# ---------------- Interactive Replay ----------------


class InteractiveReplayReal:
    """Interactive replay with WASD movement, Q/E turning, and episode navigation."""

    def __init__(
        self,
        data_dir: str,
        episode_idx: int = 0,
        fps: int = 30,
        zmq_port: int = 5556,
        mode: str = "token",
        slow_walk_speed: float = 0.3,
    ):
        self.data_dir = data_dir
        self.episode_idx = episode_idx
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.mode = mode
        self.zmq_port = zmq_port
        self.slow_walk_speed = slow_walk_speed

        # Load dataset
        self.parquet_path = resolve_episode_parquet(data_dir, episode_idx)
        self.df = pd.read_parquet(self.parquet_path)
        self.num_frames = len(self.df)
        self.current_frame_idx = 0
        print(f"[InteractiveReplayReal] Loaded: {self.parquet_path} ({self.num_frames} frames)")

        # Interactive state
        self.running = True
        self.replay_running = False

        # Facing direction for idle_planner mode (cumulative yaw)
        self.facing_yaw = 0.0
        self.facing_yaw_delta = math.pi / 12  # +/- 15 degrees

        # Movement direction for slow_walk mode
        self.movement = [0.0, 0.0, 0.0]
        self._keys_pressed: set = set()
        self._key_timestamps: dict = {}
        self._key_timeout = 0.1  # Seconds before considering key released

        # Keyboard controller
        self.kb = NonBlockingKeyboard()

        # Mode
        self.current_mode = InteractiveReplayMode.IDLE_PLANER

        # ZMQ client
        self.zmq = ReplayZMQClient(host="*", port=zmq_port, mode=mode)
        self.zmq.connect()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print(f"\n[InteractiveReplayReal] Signal {sig}, shutting down...")
        self.running = False

    def _get_frame(self, frame_idx: int):
        """Load a single frame from dataframe."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return None
        return row_to_frame(self.df.iloc[frame_idx])

    def _navigate_episode(self, delta: int):
        """Navigate to previous (delta < 0) or next (delta > 0) episode."""
        new_idx = self.episode_idx + delta
        if new_idx < 0:
            print(f"[InteractiveReplayReal] Episode {new_idx} out of range")
            return
        try:
            self.parquet_path = resolve_episode_parquet(self.data_dir, new_idx)
            self.df = pd.read_parquet(self.parquet_path)
            self.num_frames = len(self.df)
            self.episode_idx = new_idx
            self.current_frame_idx = 0
            self.replay_running = False
            print(f"[InteractiveReplayReal] Episode {new_idx} loaded ({self.num_frames} frames)")
        except (ValueError, FileNotFoundError) as e:
            print(f"[InteractiveReplayReal] Episode {new_idx} not available: {e}")

    def _send_start_planner(self):
        """Send start command to enter PLANNER mode."""
        self.zmq.send_command(start=True, stop=False, planner=True)
        print("[InteractiveReplayReal] Sent: start=True, planner=True")

    def _send_idle_planner(self):
        """Send idle planner command with current facing direction."""
        facing_x = math.cos(self.facing_yaw)
        facing_y = math.sin(self.facing_yaw)
        self.zmq.send_idle_planner(facing_x, facing_y)

    def _send_slow_walk(self):
        """Send slow walk command with current movement vector."""
        facing_x = math.cos(self.facing_yaw)
        facing_y = math.sin(self.facing_yaw)
        self.zmq.send_slow_walk(self.movement, facing_x, facing_y, self.slow_walk_speed)

    def _switch_to_idle_planner(self):
        """Switch to idle planner mode."""
        if self.current_mode != InteractiveReplayMode.IDLE_PLANER:
            self.current_mode = InteractiveReplayMode.IDLE_PLANER
            self.replay_running = False
            print("[InteractiveReplayReal] Mode: IDLE_PLANER")

    def _switch_to_slow_walk(self):
        """Switch to slow walk mode."""
        if self.current_mode != InteractiveReplayMode.SLOW_WALK:
            self.current_mode = InteractiveReplayMode.SLOW_WALK
            self.replay_running = False
            self.movement = [0.0, 0.0, 0.0]
            self._keys_pressed.clear()
            self._key_timestamps.clear()
            print("[InteractiveReplayReal] Mode: SLOW_WALK")

    def _update_movement(self):
        """Update movement vector based on currently pressed keys."""
        # Local movement (relative to facing direction)
        local_forward = 0.0  # W=forward, S=backward
        local_strafe = 0.0   # D=right, A=left
        if "w" in self._keys_pressed:
            local_forward = 1.0
        elif "s" in self._keys_pressed:
            local_forward = -1.0
        if "a" in self._keys_pressed:
            local_strafe = 1.0
        elif "d" in self._keys_pressed:
            local_strafe = -1.0

        # Turning (updates facing_yaw)
        if "q" in self._keys_pressed:
            self.facing_yaw -= self.facing_yaw_delta
        elif "e" in self._keys_pressed:
            self.facing_yaw += self.facing_yaw_delta

        # Convert local movement to world coordinates (same as official pico_manager)
        # facing = [cos(yaw), sin(yaw)]
        # rotation: world = R @ local, where R = [[-sin, cos], [cos, sin]]
        facing_x = math.cos(self.facing_yaw)
        facing_y = math.sin(self.facing_yaw)
        perp_x = -facing_y
        perp_y = facing_x

        # world_x = perp_x * local_x + facing_x * local_y
        # world_y = perp_y * local_x + facing_y * local_y
        world_x = perp_x * local_strafe + facing_x * local_forward
        world_y = perp_y * local_strafe + facing_y * local_forward

        self.movement = [world_x, world_y, 0.0]

    def _check_key_timeouts(self):
        """Check for key timeouts and release keys that haven't been refreshed."""
        if self.current_mode != InteractiveReplayMode.SLOW_WALK:
            return
        current_time = time.time()
        keys_to_release = []
        for key in self._keys_pressed:
            if key in self._key_timestamps:
                if current_time - self._key_timestamps[key] > self._key_timeout:
                    keys_to_release.append(key)
        if keys_to_release:
            for key in keys_to_release:
                self._keys_pressed.discard(key)
                self._key_timestamps.pop(key, None)
            self._update_movement()

    def _start_replay(self):
        """Start replaying the current episode."""
        if self.current_mode == InteractiveReplayMode.PLAYING:
            # Stop current replay
            self.replay_running = False
            self.current_mode = InteractiveReplayMode.IDLE_PLANER
            print("[InteractiveReplayReal] Mode: IDLE_PLANER (replay stopped)")
        else:
            # Start replay - switch to STREAMED_MOTION mode first
            if self.mode == "token":
                self.zmq.send_command(start=True, stop=False, planner=False)
                time.sleep(0.5)
            self.current_mode = InteractiveReplayMode.PLAYING
            self.replay_running = True
            self.current_frame_idx = 0
            print(f"[InteractiveReplayReal] Mode: PLAYING (episode {self.episode_idx}, {self.num_frames} frames)")

    def _handle_keyboard_input(self, key: Optional[str]):
        """Process keyboard input."""
        if key is None:
            return

        if key == "1":
            # Toggle between IDLE_PLANER and SLOW_WALK
            if self.current_mode == InteractiveReplayMode.IDLE_PLANER:
                self._switch_to_slow_walk()
            elif self.current_mode == InteractiveReplayMode.SLOW_WALK:
                self._switch_to_idle_planner()
            elif self.current_mode == InteractiveReplayMode.PLAYING:
                # Stop replay first, then toggle
                self._start_replay()

        elif key == "\n":  # Enter
            self._start_replay()

        elif key in ("z", "Z"):
            self._navigate_episode(-1)

        elif key in ("c", "C"):
            self._navigate_episode(1)

        elif key in ("q", "Q", "e", "E"):
            if self.current_mode == InteractiveReplayMode.SLOW_WALK:
                # Q/E for turning in slow walk mode
                self._keys_pressed.add(key.lower())
                self._key_timestamps[key.lower()] = time.time()
                self._update_movement()
            elif self.current_mode == InteractiveReplayMode.IDLE_PLANER:
                # Instant turn in idle mode
                if key in ("q", "Q"):
                    self.facing_yaw -= self.facing_yaw_delta
                else:
                    self.facing_yaw += self.facing_yaw_delta
                print(f"[InteractiveReplayReal] Facing: {math.degrees(self.facing_yaw):.1f} deg")

        elif key in ("w", "W", "a", "A", "s", "S", "d", "D"):
            if self.current_mode == InteractiveReplayMode.SLOW_WALK:
                self._keys_pressed.add(key.lower())
                self._key_timestamps[key.lower()] = time.time()
                self._update_movement()
            elif self.current_mode == InteractiveReplayMode.IDLE_PLANER:
                # Q/E for turning, but WASD doesn't work in IDLE mode
                pass

        elif key == "\x1b" or key == "\x03":  # ESC or Ctrl+C
            print("[InteractiveReplayReal] Quit requested")
            self.running = False

    def _print_help(self):
        """Print keyboard control help."""
        print("\n" + "=" * 50)
        print("Interactive Replay Controls:")
        print("=" * 50)
        print("  1          - Toggle IDLE_PLANER / SLOW_WALK mode")
        print("  W/A/S/D    - Move (in SLOW_WALK mode)")
        print("  Q / E      - Turn left/right")
        print("  Z          - Previous episode")
        print("  C          - Next episode")
        print("  Enter      - Play/Stop current episode")
        print("  Ctrl+C     - Exit")
        print("=" * 50 + "\n")

    def _cleanup(self):
        """Cleanup before exit."""
        print("[InteractiveReplayReal] Cleanup: switching to IDLE_PLANER...")
        # Switch to idle planner
        self.zmq.send_command(start=True, stop=False, planner=True)
        time.sleep(0.5)
        # Send idle commands for a moment
        for _ in range(30):
            self._send_idle_planner()
            time.sleep(1/30)
        # Release ZMQ
        self.zmq.release(send_stop=False)
        self.kb.stop()
        print("[InteractiveReplayReal] Cleanup complete")

    def run(self):
        """Main interactive loop."""
        print(f"[InteractiveReplayReal] Starting interactive replay, mode={self.mode}")
        self._print_help()

        # Start keyboard listener
        self.kb.start()

        # Send initial start command
        self._send_start_planner()
        time.sleep(1.0)

        # Switch to idle planner mode
        self._switch_to_idle_planner()

        prev_time = time.perf_counter()
        control_rate = 30.0  # Hz for control commands

        try:
            while self.running:
                # Process keyboard input
                while True:
                    key = self.kb.get_key()
                    if key is None:
                        break
                    self._handle_keyboard_input(key)

                # Check key timeouts
                self._check_key_timeouts()

                # Send appropriate command based on mode
                if self.current_mode == InteractiveReplayMode.IDLE_PLANER:
                    self._send_idle_planner()
                elif self.current_mode == InteractiveReplayMode.SLOW_WALK:
                    self._update_movement()
                    self._send_slow_walk()
                elif self.current_mode == InteractiveReplayMode.PLAYING:
                    if self.replay_running and self.current_frame_idx < self.num_frames:
                        frame = self._get_frame(self.current_frame_idx)
                        if frame is not None:
                            if self.mode == "token":
                                motion_token, left_hand, right_hand = extract_action_token(frame)
                                self.zmq.send_token(motion_token, left_hand, right_hand)
                            else:
                                action = extract_action_joints(frame)
                                self.zmq.send_action(action)
                            self.current_frame_idx += 1
                        else:
                            self.replay_running = False
                    else:
                        # Replay finished, switch to idle
                        self.replay_running = False
                        self.current_mode = InteractiveReplayMode.IDLE_PLANER
                        print("[InteractiveReplayReal] Replay finished, IDLE_PLANER")

                # Sleep to maintain control rate
                elapsed = time.perf_counter() - prev_time
                sleep_time = (1.0 / control_rate) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                prev_time = time.perf_counter()

        finally:
            self._cleanup()


# ---------------- Main ----------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Replay for real robot with WASD movement and episode navigation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/karthus_chen/ycb_ws/datasets/SONIC/test/tpose_halt/2026-07-26",
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Initial episode index",
    )
    parser.add_argument(
        "--zmq_port",
        type=int,
        default=5556,
        help="ZMQ PUB port for C++ WBC",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="token",
        choices=["planner", "token"],
        help='Mode: "planner" sends direct joint values, "token" sends motion_token',
    )
    parser.add_argument(
        "--slow_walk_speed",
        type=float,
        default=0.3,
        help="Speed for slow walk mode (default: 0.3)",
    )

    args = parser.parse_args()

    replay = InteractiveReplayReal(
        data_dir=args.data_dir,
        episode_idx=args.episode_idx,
        zmq_port=args.zmq_port,
        mode=args.mode,
        slow_walk_speed=args.slow_walk_speed,
    )
    try:
        replay.run()
    except KeyboardInterrupt:
        print("\n[InteractiveReplayReal] Interrupted")


if __name__ == "__main__":
    main()
