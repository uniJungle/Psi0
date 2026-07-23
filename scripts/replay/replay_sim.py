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

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Any

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
        self.sock.bind(f"tcp://{self.host}:{self.port}")
        time.sleep(0.5)
        print(f"[ReplayZMQ] Bound to tcp://{self.host}:{self.port}, mode={self.mode}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Replay a LeRobot dataset episode in MuJoCo simulation via C++ WBC."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/zzz/unitree_sh_disk/tools/ycb/datasets/SONIC/test/2026-07-22/origin/",
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=1,
        help="Episode index to replay",
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
        default="planner",
        choices=["planner", "token"],
        help='Mode: "planner" sends direct joint values, "token" sends motion_token (like real client)',
    )

    args = parser.parse_args()

    from psi.utils import resolve_data_path
    args.data_dir = str(resolve_data_path(args.data_dir))

    replay = ReplaySim(
        data_dir=args.data_dir,
        episode_idx=args.episode_idx,
        fps=args.fps,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        mode=args.mode,
    )
    replay.run()


if __name__ == "__main__":
    main()
