#!/usr/bin/env python3
"""Replay a LeRobot dataset episode on the real robot via C++ WBC ZMQ interface.

Usage:
    # 1. Start the robot-side C++ WBC controller (with zmq_manager input):
    bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

    # 2. Start PICO streamer (if using Brainco EEF):
    bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico --eef brainco --dds-interface enp4s0

    # 3. Replay data on the real robot:
    python replay_real.py --mode token --episode_idx 0 --robot_ip 192.168.123.164

    # Or if C++ WBC is running with deploy_psi0-sonic-rtc-robot.sh (input-type=manager):
    python replay_real.py --mode token --episode_idx 0 --robot_ip 192.168.123.164 --input_type manager

Architecture:
    [LeRobot Dataset]  -->  [replay_real.py]  -->  [ZMQ PUB:5556]
                                                          |
    [Real Robot + C++ WBC]  <--  [ZMQ SUB:5557]  <--------+
         (listens on ZMQ PUB:5556)

ZMQ Protocol:
    - planner mode: "planner" topic with upper_body_position (arm joints 14D)
    - token mode:   "pose" topic with token_state (64D motion token) + hand joints (14D)
    - command:      "command" topic (start/stop/planner mode)

Input Types:
    - zmq_manager: Used by collect_psi0-sonic-data-manual.sh deploy (streaming motion mode)
    - manager:     Used by deploy_psi0-sonic-rtc-robot.sh (needs explicit start/stop)
"""

from __future__ import annotations

import os
import sys
import time
import signal
import argparse
import threading
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


def extract_action_token(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract motion token and hand joints from a dataset frame."""
    # motion_token: (64D) action.motion_token
    motion_token = frame.get("action.motion_token", np.zeros(64))
    if motion_token.ndim > 1:
        motion_token = motion_token[0]

    # hand joints: (7D) teleop.left_hand_joints + (7D) teleop.right_hand_joints
    left_hand = frame.get("teleop.left_hand_joints", np.zeros(7))
    right_hand = frame.get("teleop.right_hand_joints", np.zeros(7))
    if left_hand.ndim > 1:
        left_hand = left_hand[0]
    if right_hand.ndim > 1:
        right_hand = right_hand[0]

    return motion_token, left_hand, right_hand


def extract_action_joints(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract joint values from a dataset frame (for planner mode)."""
    # observation.state contains all joint positions (43D)
    state = frame.get("observation.state", np.zeros(43))
    if state.ndim > 1:
        state = state[0]

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


# ---------------- ZMQ Client ----------------


class ReplayZMQClient:
    """ZMQ client for sending replay commands to C++ WBC."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
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
        """Connect to ZMQ publisher."""
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        endpoint = f"tcp://{self.host}:{self.port}"
        self.sock.connect(endpoint)

        # Give subscriber time to connect
        time.sleep(0.5)
        if self.verbose:
            print(f"[ReplayZMQ] Connected to {endpoint}, mode={self.mode}, input_type={self.input_type}")

    def send_command(self, start: bool = False, stop: bool = False, planner: bool = True):
        """Send command message to switch modes."""
        topic = b"command"
        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]

        header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
        header_json = json.dumps(header).encode("utf-8")
        HEADER_SIZE = 1280
        header_bytes = header_json + b"\x00" * (HEADER_SIZE - len(header_json))

        data = b""
        data += struct.pack("B", 1 if start else 0)
        data += struct.pack("B", 1 if stop else 0)
        data += struct.pack("B", 1 if planner else 0)

        message = topic + header_bytes + data
        self.sock.send(message)

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
        if left_hand.ndim > 1:
            left_hand = left_hand[0]
        if right_hand.ndim > 1:
            right_hand = right_hand[0]

        # FSQ quantize the motion token
        token_qtz = fsq_quantize(motion_token)

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
        """Stop and cleanup."""
        if self.sock:
            self.sock.close(linger=0)
        if self.ctx:
            self.ctx.term()
        print("[ReplayZMQ] Stopped")


# ---------------- Main Replay ----------------


class ReplayReal:
    """Replay a dataset episode on the real robot."""

    def __init__(
        self,
        data_dir: str,
        episode_idx: int = 0,
        fps: int = 30,
        robot_ip: str = "192.168.123.164",
        zmq_port: int = 5556,
        mode: str = "token",
        input_type: str = "zmq_manager",
        warmup_seconds: float = 2.0,
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
        """
        self.data_dir = data_dir
        self.episode_idx = episode_idx
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.mode = mode
        self.input_type = input_type
        self.warmup_seconds = warmup_seconds
        self.running = True

        # Load dataset (local only, no HuggingFace)
        from psi.data.lerobot.compat import LeRobotDataset

        data_path = Path(data_dir)
        repo_id = data_path.name
        root = str(data_path)
        self.full_dataset = LeRobotDataset(repo_id=repo_id, root=root)
        print(f"[ReplayReal] Loaded dataset: repo_id={repo_id}, root={root}, episodes={self.full_dataset.num_episodes}, total_frames={len(self.full_dataset)}")

        # Select specific episode
        episode_index = self.full_dataset.episode_data_index
        num_episodes = self.full_dataset.num_episodes

        if episode_idx >= num_episodes:
            raise ValueError(f"Episode index {episode_idx} out of range, available: 0-{num_episodes-1}")

        start_idx = episode_index["from"][episode_idx].item()
        end_idx = episode_index["to"][episode_idx].item()
        self.episode_indices = list(range(start_idx, end_idx))
        print(f"[ReplayReal] Selected episode {self.episode_idx}: frames {start_idx}-{end_idx-1} ({len(self.episode_indices)} frames)")

        # Preload all frames into memory
        print(f"[ReplayReal] Preloading {len(self.episode_indices)} frames into memory...")
        preload_start = time.perf_counter()
        self.frames = []
        for idx in self.episode_indices:
            frame = self.full_dataset[idx]
            frame_data = {}
            for key, value in frame.items():
                if hasattr(value, 'numpy'):
                    frame_data[key] = value.numpy()
                else:
                    frame_data[key] = np.asarray(value)
            self.frames.append(frame_data)
        preload_time = time.perf_counter() - preload_start
        print(f"[ReplayReal] Preloading done in {preload_time:.2f}s ({preload_time/len(self.frames)*1000:.1f}ms per frame)")

        # ZMQ client
        # Note: For real robot, the C++ WBC runs on the robot and listens on a port.
        # The replay script runs on the workstation and connects as a publisher.
        # Use localhost if running on the same machine, or robot's IP otherwise.
        zmq_host = robot_ip if robot_ip != "localhost" else "localhost"
        self.zmq = ReplayZMQClient(
            host=zmq_host,
            port=zmq_port,
            mode=mode,
            input_type=input_type,
        )
        self.zmq.connect()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print(f"\n[ReplayReal] Signal {sig}, shutting down...")
        self.running = False

    def run(self):
        """Run the replay."""
        print(f"[ReplayReal] Starting replay at {self.fps} Hz, mode={self.mode}, input_type={self.input_type}")

        # Determine planner mode flag
        if self.mode == "token":
            planner_mode = False  # streamed motion mode expects pose topic
        else:
            planner_mode = True   # planner mode expects planner topic

        # Send start command
        # For zmq_manager: C++ WBC auto-starts when it receives pose messages
        # For manager: need explicit start command
        if self.input_type == "manager":
            self.zmq.send_command(start=True, stop=False, planner=planner_mode)
            print(f"[ReplayReal] Waiting {self.warmup_seconds}s for warmup...")
            time.sleep(self.warmup_seconds)

        frame_idx = 0
        prev_time = time.perf_counter()
        started = False

        while self.running and frame_idx < len(self.frames):
            frame = self.frames[frame_idx]

            # For zmq_manager mode, first pose triggers start
            if self.input_type == "zmq_manager" and not started:
                self.zmq.send_command(start=True, stop=False, planner=planner_mode)
                started = True
                print(f"[ReplayReal] Sent start command, beginning replay...")

            # Send based on mode
            if self.mode == "token":
                motion_token, left_hand, right_hand = extract_action_token(frame)
                self.zmq.send_token(motion_token, left_hand, right_hand)
            else:
                action = extract_action_joints(frame)
                self.zmq.send_action(action)

            # Progress logging
            if frame_idx % 30 == 0:
                print(f"[ReplayReal] Frame {frame_idx}/{len(self.frames)}")

            # Frame timing
            elapsed = time.perf_counter() - prev_time
            sleep_time = self.frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            prev_time = time.perf_counter()
            frame_idx += 1

        # Replay finished
        print("[ReplayReal] Replay finished")

        # For manager mode, send stop command
        if self.input_type == "manager":
            print("[ReplayReal] Sending stop command...")
            self.zmq.send_command(start=False, stop=True, planner=planner_mode)

        print("[ReplayReal] Waiting, press Ctrl+C to exit")
        while self.running:
            time.sleep(1)

        print("[ReplayReal] Shutting down...")


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
        default="/home/zzz/unitree_sh_disk/tools/ycb/datasets/SONIC/test/2026-07-22/origin",
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
        help="Robot IP address (default: 192.168.123.164)",
    )
    parser.add_argument(
        "--zmq_port",
        type=int,
        default=5556,
        help="ZMQ port (default: 5556)",
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
    )
    replay.run()


if __name__ == "__main__":
    # Import zmq and json at top level
    import zmq
    import json
    import struct
    main()
