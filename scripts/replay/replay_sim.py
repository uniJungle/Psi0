#!/usr/bin/env python3
"""Replay a LeRobot dataset episode in MuJoCo simulation via C++ WBC ZMQ interface.

Usage:
    # Mode 1: Direct joint values (via planner topic)
    python replay_sim.py --mode planner --episode_idx 0

    # Mode 2: Motion token (via pose topic, like real client)
    python replay_sim.py --mode token --episode_idx 0

Architecture:
    [LeRobot Dataset]  -->  [replay_sim.py]  -->  [ZMQ PUB:5556]
                                                          |
    [MuJoCo + C++ WBC sim]  <--  [ZMQ SUB:5557]  <--------+
         (listens on ZMQ PUB:5556)

ZMQ Protocol:
    - planner mode: "planner" topic with upper_body_position (arm joints 14D)
    - token mode:   "pose" topic with token_state (64D motion token) + hand joints (14D)
    - command:      "command" topic (start/stop/planner mode)
"""

from __future__ import annotations

import math
import os
import queue
import sys
import termios
import threading
import time
import signal
import argparse
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add third_party/GR00T-WholeBodyControl to path for imports
_THIRD_PARTY = Path(__file__).parent.parent.parent / "third_party" / "GR00T-WholeBodyControl"
sys.path.insert(0, str(_THIRD_PARTY))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)


# ---------------- FSQ Quantization (for motion token) ----------------
# Match the settings in psi_rtc_sonic_client.py
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized


# ---------------- Joint Definitions ----------------
# Match the definitions in scripts/viz/g1.py and scripts/viz/viz_episode_real.py

# Hand joint order matches info.json: index -> middle -> thumb
HAND_JOINT_NAMES = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


# ---------------- Action Extraction ----------------

def extract_action_joints(frame: dict[str, Any]) -> dict[str, float]:
    """Extract action joints from a dataset frame.

    The dataset uses 'action.wbc' field with 43D vector (from info.json):
        action[0:12]  -> leg joints (12D, left+right legs)
        action[12:15] -> waist joints (3D)
        action[15:22] -> left arm joints (7D)
        action[22:29] -> left hand joints (7D)
        action[29:36] -> right arm joints (7D)
        action[36:43] -> right hand joints (7D)
    """
    action_wbc = frame["action.wbc"]
    if hasattr(action_wbc, 'numpy'):
        action_np = action_wbc.numpy()
    else:
        action_np = np.asarray(action_wbc)

    action = {}
    # Legs: zeroed for WBC (indices 0-14 in LEG_JOINT_NAMES)
    action.update(dict(zip(LEG_JOINT_NAMES[:12], [0.0] * 12)))  # 12 leg joints
    action.update(dict(zip(LEG_JOINT_NAMES[12:15], [0.0] * 3)))  # 3 waist joints

    # Left arm: ARM_JOINT_NAMES[0:7] = left arm joints
    action.update(dict(zip(ARM_JOINT_NAMES[:7], action_np[15:22].tolist())))

    # Left hand: HAND_JOINT_NAMES[0:7] = left hand joints
    action.update(dict(zip(HAND_JOINT_NAMES[:7], action_np[22:29].tolist())))

    # Right arm: ARM_JOINT_NAMES[7:14] = right arm joints
    action.update(dict(zip(ARM_JOINT_NAMES[7:14], action_np[29:36].tolist())))

    # Right hand: HAND_JOINT_NAMES[7:14] = right hand joints
    action.update(dict(zip(HAND_JOINT_NAMES[7:14], action_np[36:43].tolist())))

    return action


def extract_action_token(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract motion token and hand joints from a dataset frame.

    Returns:
        motion_token: (64D) motion token from action.motion_token
        left_hand: (7D) left hand joints
        right_hand: (7D) right hand joints
    """
    motion_token = frame["action.motion_token"]
    if hasattr(motion_token, 'numpy'):
        motion_token = motion_token.numpy()
    else:
        motion_token = np.asarray(motion_token)

    left_hand = frame["teleop.left_hand_joints"]
    if hasattr(left_hand, 'numpy'):
        left_hand = left_hand.numpy()
    else:
        left_hand = np.asarray(left_hand)

    right_hand = frame["teleop.right_hand_joints"]
    if hasattr(right_hand, 'numpy'):
        right_hand = right_hand.numpy()
    else:
        right_hand = np.asarray(right_hand)

    return motion_token, left_hand, right_hand


def action_to_planner_fields(action: dict[str, float]) -> dict:
    """Convert joint action dict to planner message fields.

    Note: ARM_JOINT_NAMES[0:7] = left arm, ARM_JOINT_NAMES[7:14] = right arm
          HAND_JOINT_NAMES[0:7] = left hand, HAND_JOINT_NAMES[7:14] = right hand
    """
    # Only replay arms, set hands to fixed default (open) position
    upper_body = (
        [action[name] for name in ARM_JOINT_NAMES[0:7]] +  # left arm (7D)
        [action[name] for name in ARM_JOINT_NAMES[7:14]]   # right arm (7D)
    )  # total 14D

    # Fixed hand positions (open hand = 0.0 radians)
    left_hand = [0.0] * 7   # 7D left hand fixed
    right_hand = [0.0] * 7  # 7D right hand fixed

    return {
        "upper_body_position": upper_body,  # 14D arm
        "left_hand_position": left_hand,   # 7D left hand fixed
        "right_hand_position": right_hand,  # 7D right hand fixed
    }


# ---------------- ZMQ Sender ----------------

class ReplayZMQClient:
    """Send action commands to C++ WBC via ZMQ."""

    def __init__(self, host: str = "localhost", port: int = 5556, mode: str = "planner"):
        """
        Args:
            host: ZMQ bind address
            port: ZMQ port
            mode: "planner" for upper_body_position, "token" for motion_token
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.ctx = None
        self.sock = None
        self._frame_index = 0

    def connect(self):
        import zmq
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        # Bind to all interfaces for simulation (C++ WBC connects via localhost)
        bind_host = self.host if self.host not in ("localhost", "127.0.0.1") else "*"
        self.sock.bind(f"tcp://{bind_host}:{self.port}")
        time.sleep(0.5)
        print(f"[ReplayZMQ] Bound to tcp://{bind_host}:{self.port}, mode={self.mode}")

    def send_command(self, start: bool = False, stop: bool = False, planner: bool = True):
        """Send control command (start/stop/planner mode)."""
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self.sock.send(msg)
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
        Send motion token via both planner and pose topics.

        The planner=True command tells C++ WBC to use external input.
        We send:
        1. planner topic: empty or minimal to keep connection alive
        2. pose topic: token_state (64D) + hand joints (14D)

        Args:
            motion_token: (64D) quantized motion token
            left_hand: (7D) left hand joints
            right_hand: (7D) right hand joints
        """
        if motion_token.ndim > 1:
            motion_token = motion_token[0]
        if left_hand.ndim > 1:
            left_hand = left_hand[0]
        if right_hand.ndim > 1:
            right_hand = right_hand[0]

        # FSQ quantize the motion token
        token_qtz = fsq_quantize(motion_token)

        # === Send via pose topic (Protocol v4) ===
        # Build pose message: hand_joints(14) + token(64)
        action_out = np.concatenate([token_qtz, left_hand, right_hand]).astype(np.float32)
        pose_data = {
            "token_state": action_out[np.newaxis, :64],       # (1, 64)
            "left_hand_joints": action_out[np.newaxis, 64:71],    # (1, 7)
            "right_hand_joints": action_out[np.newaxis, 71:78],   # (1, 7)
        }
        pose_msg = pack_pose_message(pose_data, topic="pose", version=4)
        self.sock.send(pose_msg)

        self._frame_index += 1

    def stop(self):
        if self.sock:
            self.sock.close(linger=0)
        if self.ctx:
            self.ctx.term()
        print("[ReplayZMQ] Stopped")


# ---------------- Main Replay ----------------

class ReplaySim:
    """Replay a dataset episode in simulation."""

    def __init__(
        self,
        data_dir: str,
        episode_idx: int = 0,
        fps: int = 30,
        zmq_host: str = "localhost",
        zmq_port: int = 5556,
        mode: str = "planner",
    ):
        """
        Args:
            data_dir: Path to LeRobot dataset directory
            episode_idx: Episode index to replay
            fps: Target replay FPS
            zmq_host: ZMQ host
            zmq_port: ZMQ port
            mode: "planner" for direct joint values, "token" for motion_token
        """
        self.data_dir = data_dir
        self.episode_idx = episode_idx
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.mode = mode
        self.running = True

        # Load dataset (local only, no HuggingFace)
        from psi.data.lerobot.compat import LeRobotDataset

        data_path = Path(data_dir)
        repo_id = data_path.name  # e.g., "clean"
        root = str(data_path)  # point root to the dataset dir itself
        self.full_dataset = LeRobotDataset(repo_id=repo_id, root=root)
        print(f"[ReplaySim] Loaded dataset: repo_id={repo_id}, root={root}, episodes={self.full_dataset.num_episodes}, total_frames={len(self.full_dataset)}")

        # Select specific episode
        episode_index = self.full_dataset.episode_data_index
        num_episodes = self.full_dataset.num_episodes

        if episode_idx >= num_episodes:
            raise ValueError(f"Episode index {episode_idx} out of range, available: 0-{num_episodes-1}")

        # episode_data_index is a dict with 'from' and 'to' tensors
        start_idx = episode_index["from"][episode_idx].item()
        end_idx = episode_index["to"][episode_idx].item()
        self.episode_indices = list(range(start_idx, end_idx))
        print(f"[ReplaySim] Selected episode {self.episode_idx}: frames {start_idx}-{end_idx-1} ({len(self.episode_indices)} frames)")

        # Preload all frames into memory to avoid slow per-frame loading
        print(f"[ReplaySim] Preloading {len(self.episode_indices)} frames into memory...")
        preload_start = time.perf_counter()
        self.frames = []
        for idx in self.episode_indices:
            frame = self.full_dataset[idx]
            # Convert tensors to numpy for faster access
            frame_data = {}
            for key, value in frame.items():
                if hasattr(value, 'numpy'):
                    frame_data[key] = value.numpy()
                else:
                    frame_data[key] = np.asarray(value)
            self.frames.append(frame_data)
        preload_time = time.perf_counter() - preload_start
        print(f"[ReplaySim] Preloading done in {preload_time:.2f}s ({preload_time/len(self.frames)*1000:.1f}ms per frame)")

        # ZMQ client (pass mode to determine topic)
        self.zmq = ReplayZMQClient(host=zmq_host, port=zmq_port, mode=mode)
        self.zmq.connect()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print(f"\n[ReplaySim] Signal {sig}, shutting down...")
        self.running = False

    def run(self):
        print(f"[ReplaySim] Starting replay at {self.fps} Hz, mode={self.mode}")

        # Send start command based on mode
        # planner=True: use planner topic (direct joint values)
        # planner=False: use pose topic (motion token) - streamed motion mode
        if self.mode == "token":
            planner_mode = False  # streamed motion mode expects pose topic
        else:
            planner_mode = True   # planner mode expects planner topic
        self.zmq.send_command(start=True, stop=False, planner=planner_mode)

        frame_idx = 0
        prev_time = time.perf_counter()

        while self.running and frame_idx < len(self.frames):
            # Use preloaded frame (numpy arrays, no loading overhead)
            frame = self.frames[frame_idx]

            # Send based on mode
            if self.mode == "token":
                motion_token, left_hand, right_hand = extract_action_token(frame)
                self.zmq.send_token(motion_token, left_hand, right_hand)
            else:
                action = extract_action_joints(frame)
                self.zmq.send_action(action)

            # Progress logging
            if frame_idx % 30 == 0:
                print(f"[ReplaySim] Frame {frame_idx}/{len(self.frames)}")

            # Frame timing
            elapsed = time.perf_counter() - prev_time
            sleep_time = self.frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            prev_time = time.perf_counter()

            frame_idx += 1

        # Replay finished - stay in IDLE mode, don't send stop command
        print("[ReplaySim] Replay finished, waiting in IDLE mode...")
        print("[ReplaySim] Press Ctrl+C to exit")

        # Keep connection alive, C++ WBC will timeout and go to IDLE
        while self.running:
            time.sleep(1)

        print("[ReplaySim] Shutting down...")


# ---------------- Keyboard Controller ----------------


class KeyboardController:
    """Terminal keyboard input listener (non-blocking, independent thread)."""

    # No arrow keys - using z/c for episode navigation

    def __init__(self):
        self._running = True
        self._key_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._original_termios: Optional[tuple] = None

    def _setup_terminal(self):
        """Set terminal to non-canonical, non-blocking mode."""
        self._original_termios = termios.tcgetattr(sys.stdin)
        new_attrs = termios.tcgetattr(sys.stdin)
        # ECHO: don't echo input
        # ICANON: non-canonical mode (no line buffering)
        # ISIG: don't generate SIGINT on ctrl-c
        new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON | termios.ISIG)
        new_attrs[6][termios.VMIN] = 0  # non-blocking read
        new_attrs[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin, termios.TCSANOW, new_attrs)

    def _restore_terminal(self):
        """Restore original terminal settings."""
        if self._original_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, self._original_termios)

    def _read_loop(self):
        """Background thread that reads keyboard input."""
        import select
        while self._running:
            try:
                # Use select to wait for input with timeout
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._key_queue.put(ch)
            except Exception:
                break
            time.sleep(0.01)  # Small sleep to avoid busy waiting

    def start(self):
        """Start the keyboard listener thread."""
        self._setup_terminal()
        self._drain_input()  # Clear any buffered input
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[Keyboard] Control enabled")

    def _drain_input(self):
        """Drain any buffered input from stdin."""
        import select
        while select.select([sys.stdin], [], [], 0.01)[0]:
            sys.stdin.read(10)  # Read up to 10 chars at a time

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


class InteractiveReplaySim:
    """Interactive simulation replay with keyboard control."""

    # Locomotion mode constants (from C++ enum)
    LOCOMOTION_MODE_IDLE = 0
    LOCOMOTION_MODE_SLOW_WALK = 2  # walking mode for keyboard control

    def __init__(
        self,
        data_dir: str,
        episode_idx: int = 0,
        fps: int = 30,
        zmq_host: str = "localhost",
        zmq_port: int = 5556,
        mode: str = "token",
        slow_walk_speed: float = 0.3,
        turn_speed: float = 1.0,
        data_dirs: Optional[list[str]] = None,
    ):
        """
        Args:
            data_dir: Path to LeRobot dataset directory
            episode_idx: Initial episode index
            fps: Target replay FPS
            zmq_host: ZMQ host
            zmq_port: ZMQ port
            mode: "planner" or "token"
            slow_walk_speed: Speed for slow walk mode
            data_dirs: Optional list of data directories for navigation
        """
        self.data_dir = data_dir
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.mode = mode
        self.slow_walk_speed = slow_walk_speed
        self.turn_speed = turn_speed
        self.data_dirs = data_dirs or [data_dir]

        # State
        self.current_mode = InteractiveReplayMode.IDLE_PLANER
        self.current_episode_idx = episode_idx
        self.num_episodes = 0
        self.running = True
        self.replay_running = False
        self.frames: list[dict[str, Any]] = []
        self.current_frame_idx = 0

        # Facing direction for idle_planner mode (cumulative yaw)
        self.facing_yaw = 0.0
        self.facing_yaw_delta = math.pi / 12  # +/- 15 degrees

        # Movement direction for slow_walk mode (reset when keys released)
        self.movement = [0.0, 0.0, 0.0]  # [forward, strafe, turn]
        self._keys_pressed: set = set()  # Track currently pressed keys
        self._key_timestamps: dict = {}  # Track when each key was last pressed
        self._key_timeout = 0.1  # Seconds before considering key released

        # Keyboard controller
        self.kb = KeyboardController()

        # Load all episodes metadata (just count, no frame data)
        self._load_episodes_metadata()

        # Initialize episode state (lazy load: only store metadata)
        self.current_episode_idx = -1
        self.episode_indices = []
        self.episode_dataset = None  # Store dataset reference for lazy loading
        self.episode_data_dir = None

        # ZMQ client
        from psi.data.lerobot.compat import LeRobotDataset
        self._lerobot_dataset_class = LeRobotDataset
        self.zmq = ReplayZMQClient(host=zmq_host, port=zmq_port, mode=mode)
        self.zmq.connect()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_episodes_metadata(self):
        """Load metadata for all episodes across all data directories."""
        from psi.data.lerobot.compat import LeRobotDataset

        episode_count = 0
        for data_dir in self.data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                print(f"[InteractiveReplay] Warning: data_dir not found: {data_dir}")
                continue
            try:
                repo_id = data_path.name
                dataset = LeRobotDataset(repo_id=repo_id, root=str(data_path))
                episode_count += dataset.num_episodes
            except Exception as e:
                print(f"[InteractiveReplay] Warning: failed to load {data_dir}: {e}")
        self.num_episodes = episode_count
        print(f"[InteractiveReplay] Total episodes available: {self.num_episodes}")

    def _load_episode(self, idx: int):
        """Load episode metadata (indices only), frames are loaded on-demand."""
        from psi.data.lerobot.compat import LeRobotDataset

        if idx < 0 or idx >= self.num_episodes:
            print(f"[InteractiveReplay] Episode {idx} out of range (0-{self.num_episodes - 1})")
            return False

        # Find which data_dir contains this episode
        cumulative = 0
        for data_dir in self.data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                continue
            try:
                repo_id = data_path.name
                dataset = LeRobotDataset(repo_id=repo_id, root=str(data_path))
                if idx < cumulative + dataset.num_episodes:
                    local_idx = idx - cumulative
                    episode_index = dataset.episode_data_index
                    start_idx = episode_index["from"][local_idx].item()
                    end_idx = episode_index["to"][local_idx].item()
                    self.episode_indices = list(range(start_idx, end_idx))
                    self.episode_dataset = dataset
                    self.episode_data_dir = data_dir
                    self.current_episode_idx = idx
                    print(f"[InteractiveReplay] Episode {idx} ready ({len(self.episode_indices)} frames)")
                    return True
                cumulative += dataset.num_episodes
            except Exception as e:
                print(f"[InteractiveReplay] Warning: failed to load {data_dir}: {e}")
                continue

        return False

    def _get_frame(self, frame_idx: int):
        """Load a single frame on-demand from dataset."""
        if self.episode_dataset is None:
            return None
        try:
            global_idx = self.episode_indices[frame_idx]
            frame = self.episode_dataset[global_idx]
            frame_data = {}
            for key, value in frame.items():
                if hasattr(value, 'numpy'):
                    frame_data[key] = value.numpy()
                else:
                    frame_data[key] = np.asarray(value)
            return frame_data
        except Exception as e:
            print(f"[InteractiveReplay] Error loading frame {frame_idx}: {e}")
            return None

    def _navigate_episode(self, delta: int):
        """Navigate to previous (delta < 0) or next (delta > 0) episode."""
        new_idx = self.current_episode_idx + delta
        if new_idx < 0 or new_idx >= self.num_episodes:
            print(f"[InteractiveReplay] Episode {new_idx} out of range")
            return
        self._load_episode(new_idx)
        print(f"[InteractiveReplay] Now on episode {self.current_episode_idx}")

    def _signal_handler(self, sig, frame):
        print(f"\n[InteractiveReplay] Signal {sig}, shutting down...")
        self.running = False

    def _send_start_planner(self):
        """Send start command to enter PLANNER mode."""
        msg = build_command_message(start=True, stop=False, planner=True)
        self.zmq.sock.send(msg)
        print("[InteractiveReplay] Sent: start=True, planner=True")

    def _send_idle_planner(self):
        """Send idle planner command (robot stands still, facing direction adjustable)."""
        # Use facing vector for direction control
        facing_x = math.cos(self.facing_yaw)
        facing_y = math.sin(self.facing_yaw)
        msg = build_planner_message(
            mode=self.LOCOMOTION_MODE_IDLE,
            movement=[0.0, 0.0, 0.0],
            facing=[facing_x, facing_y, 0.0],
            speed=-1.0,
            height=-1.0,
        )
        self.zmq.sock.send(msg)

    def _send_slow_walk(self):
        """Send slow walk command with current movement vector."""
        facing_x = math.cos(self.facing_yaw)
        facing_y = math.sin(self.facing_yaw)
        if self.current_frame_idx == 0:  # Only log once
            print(f"[DEBUG] _send_slow_walk: movement={self.movement}, facing=[{facing_x:.2f}, {facing_y:.2f}]")
        msg = build_planner_message(
            mode=self.LOCOMOTION_MODE_SLOW_WALK,
            movement=[self.movement[0], self.movement[1], 0.0],  # [forward, strafe, 0]
            facing=[facing_x, facing_y, 0.0],  # facing direction
            speed=self.slow_walk_speed,
            height=-1.0,
        )
        self.zmq.sock.send(msg)

    def _switch_to_idle_planner(self):
        """Switch to idle planner mode."""
        if self.current_mode != InteractiveReplayMode.IDLE_PLANER:
            self.current_mode = InteractiveReplayMode.IDLE_PLANER
            self.replay_running = False
            print("[InteractiveReplay] Mode: IDLE_PLANER")

    def _switch_to_slow_walk(self):
        """Switch to slow walk mode."""
        if self.current_mode != InteractiveReplayMode.SLOW_WALK:
            self.current_mode = InteractiveReplayMode.SLOW_WALK
            self.replay_running = False
            self.movement = [0.0, 0.0, 0.0]
            self._keys_pressed.clear()
            self._key_timestamps.clear()
            print("[InteractiveReplay] Mode: SLOW_WALK")

    def _update_movement(self):
        """Update movement vector based on currently pressed keys."""
        # Local movement (relative to facing direction)
        local_forward = 0.0  # ly: W=forward, S=backward
        local_strafe = 0.0   # lx: D=right, A=left
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
            self.facing_yaw -= self.turn_speed * 0.033
        elif "e" in self._keys_pressed:
            self.facing_yaw += self.turn_speed * 0.033
        
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
            print("[InteractiveReplay] Mode: IDLE_PLANER (replay stopped)")
        else:
            # Lazy load: if episode not loaded yet, load metadata first
            if self.current_episode_idx < 0 or not self.episode_indices:
                print("[InteractiveReplay] Loading episode metadata...")
                if not self._load_episode(0 if self.current_episode_idx < 0 else self.current_episode_idx):
                    print("[InteractiveReplay] Failed to load episode, cannot start replay")
                    return

            # Start replay - switch to STREAMED_MOTION mode first
            if self.mode == "token":
                self.zmq.send_command(start=True, stop=False, planner=False)
                time.sleep(0.5)
            self.current_mode = InteractiveReplayMode.PLAYING
            self.replay_running = True
            self.current_frame_idx = 0
            print(f"[InteractiveReplay] Mode: PLAYING (episode {self.current_episode_idx}, {len(self.episode_indices)} frames)")

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
                print(f"[InteractiveReplay] Facing: {math.degrees(self.facing_yaw):.1f} deg")

        elif key in ("w", "W", "a", "A", "s", "S", "d", "D"):
            print(f"[DEBUG] WASD key: {repr(key)}, mode: {self.current_mode}")
            if self.current_mode == InteractiveReplayMode.SLOW_WALK:
                self._keys_pressed.add(key.lower())
                self._key_timestamps[key.lower()] = time.time()
                self._update_movement()
                print(f"[DEBUG] movement: {self.movement}")
            elif self.current_mode == InteractiveReplayMode.IDLE_PLANER:
                # Q/E for turning, but WASD doesn't work in IDLE mode
                print("[DEBUG] WASD ignored - not in SLOW_WALK mode")
                pass

        elif key == "\x1b" or key == "\x03":  # ESC or Ctrl+C
            print("[InteractiveReplay] Quit requested")
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

    def run(self):
        """Main interactive loop."""
        print(f"[InteractiveReplay] Starting interactive replay, mode={self.mode}")
        print(f"[InteractiveReplay] Total episodes: {self.num_episodes}")
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
        control_interval = 1.0 / control_rate

        try:
            while self.running:
                # Process keyboard input
                key = self.kb.get_key()
                while key is not None:
                    self._handle_keyboard_input(key)
                    if not self.running:
                        break
                    key = self.kb.get_key()

                if not self.running:
                    break

                # Check for key timeouts (key released detection)
                self._check_key_timeouts()

                # Send control commands based on mode
                if self.current_mode == InteractiveReplayMode.IDLE_PLANER:
                    self._send_idle_planner()
                elif self.current_mode == InteractiveReplayMode.SLOW_WALK:
                    self._send_slow_walk()
                elif self.current_mode == InteractiveReplayMode.PLAYING:
                    # Replay loop (lazy load frames)
                    if self.replay_running and self.current_frame_idx < len(self.episode_indices):
                        frame = self._get_frame(self.current_frame_idx)
                        if frame is None:
                            print("[InteractiveReplay] Failed to load frame, stopping")
                            self.replay_running = False
                        else:
                            # Send based on mode
                            if self.mode == "token":
                                motion_token, left_hand, right_hand = extract_action_token(frame)
                                self.zmq.send_token(motion_token, left_hand, right_hand)
                            else:
                                action = extract_action_joints(frame)
                                self.zmq.send_action(action)

                            # Progress logging
                            if self.current_frame_idx % 30 == 0:
                                print(f"[InteractiveReplay] Frame {self.current_frame_idx}/{len(self.episode_indices)}")

                            self.current_frame_idx += 1
                    else:
                        # Replay finished
                        if self.replay_running:
                            print("[InteractiveReplay] Replay finished")
                            self.replay_running = False
                        self.current_mode = InteractiveReplayMode.IDLE_PLANER
                        print("[InteractiveReplay] Mode: IDLE_PLANER (replay done)")

                # Frame timing
                elapsed = time.perf_counter() - prev_time
                sleep_time = control_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                prev_time = time.perf_counter()

        finally:
            print("[InteractiveReplay] Shutting down...")
            # Send idle planner before exit
            self._switch_to_idle_planner()
            for _ in range(10):  # Send a few frames
                self._send_idle_planner()
                time.sleep(0.033)
            self.kb.stop()
            self.zmq.stop()
            print("[InteractiveReplay] Done")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive replay with keyboard control for MuJoCo simulation via C++ WBC."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/zzz/zzy/walk_to_table_and_place_apple_on_pink_plate",
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=None,
        help="List of data directories for episode navigation",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Initial episode index",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Replay frame rate (Hz). Dataset fps is typically 30.",
    )
    parser.add_argument(
        "--zmq_host",
        type=str,
        default="localhost",
        help="ZMQ host for C++ WBC",
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
        help='Mode: "planner" sends direct joint values, "token" sends motion_token (like real client)',
    )
    parser.add_argument(
        "--slow_walk_speed",
        type=float,
        default=0.3,
        help="Speed for slow walk mode (default: 0.3)",
    )

    args = parser.parse_args()

    from psi.utils import resolve_data_path
    data_dir = str(resolve_data_path(args.data_dir))
    data_dirs = [str(resolve_data_path(d)) for d in args.data_dirs] if args.data_dirs else [data_dir]

    replay = InteractiveReplaySim(
        data_dir=data_dir,
        episode_idx=args.episode_idx,
        fps=args.fps,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        mode=args.mode,
        slow_walk_speed=args.slow_walk_speed,
        data_dirs=data_dirs,
    )
    try:
        replay.run()
    except KeyboardInterrupt:
        print("\n[InteractiveReplay] Interrupted")
        replay.running = False


if __name__ == "__main__":
    main()
