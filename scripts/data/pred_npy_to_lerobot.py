#!/usr/bin/env python3
"""Convert closed-loop ACT ``pred_actions.npy`` to lerobot_v2.1 for ``replay_real.py``.

Input (from ``act_inference.py --save-pred-action``)::

    <pred_dir>/pred_actions.npy   # (T, 68) = token64 + hand4
    <pred_dir>/pred_states.npy    # optional (T, 29) body qpos

Output (same layout as ``openloop_act_g1_real.py``)::

    <out_dir>/
      meta/info.json ...
      data/chunk-000/episode_000000.parquet

Replay::

    python scripts/replay/replay_real.py \\
      --input_type zmq_manager --mode token --eef brainco \\
      --data_dir <out_dir> --episode_idx 0
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATE_DIM = 33
QPOS_DIM = 29
ACTION_DIM = 68
TOKEN_DIM = 64
HAND_DIM = 2
FPS_DEFAULT = 30


def _pad_states(states: np.ndarray | None, n: int) -> np.ndarray:
    """Build (T, 33) states: qpos29 + hand4. Missing dims / file → zeros."""
    out = np.zeros((n, STATE_DIM), dtype=np.float32)
    if states is None:
        return out
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"pred_states must be 2D, got {states.shape}")
    if states.shape[0] != n:
        raise ValueError(f"pred_states T={states.shape[0]} != pred_actions T={n}")
    d = min(states.shape[1], STATE_DIM)
    out[:, :d] = states[:, :d]
    return out


def write_episode_parquet(
    out_parquet: Path,
    states: np.ndarray,
    actions: np.ndarray,
    fps: float,
    episode_index: int = 0,
    task_index: int = 0,
) -> None:
    n = states.shape[0]
    rows = []
    for i in range(n):
        act = actions[i].astype(np.float32)
        rows.append(
            {
                "states": states[i].astype(np.float32).tolist(),
                "action": act.tolist(),
                "action.motion_token": act[:TOKEN_DIM].tolist(),
                "teleop.left_hand_joints": act[TOKEN_DIM : TOKEN_DIM + HAND_DIM].tolist(),
                "teleop.right_hand_joints": act[
                    TOKEN_DIM + HAND_DIM : TOKEN_DIM + 2 * HAND_DIM
                ].tolist(),
                "timestamp": float(i / fps),
                "frame_index": int(i),
                "episode_index": int(episode_index),
                "index": int(i),
                "task_index": int(task_index),
                "next.done": bool(i == n - 1),
            }
        )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)


def write_meta(
    out_dir: Path,
    n_frames: int,
    instruction: str,
    fps: int,
    template_info: dict[str, Any] | None,
    output_dir: Path | None,
) -> None:
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if template_info is not None:
        features = dict(template_info.get("features", {}))
        robot_type = template_info.get("robot_type", "g1")
        chunks_size = int(template_info.get("chunks_size", 1000))
        codebase_version = template_info.get("codebase_version", "v2.1")
    else:
        features = {
            "states": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }
        robot_type = "g1"
        chunks_size = 1000
        codebase_version = "v2.1"

    # Drop video keys: closed-loop pred has no synced mp4 by default.
    features = {
        k: v
        for k, v in features.items()
        if not (isinstance(v, dict) and v.get("dtype") == "video")
    }
    features["action.motion_token"] = {"dtype": "float32", "shape": [TOKEN_DIM]}
    features["teleop.left_hand_joints"] = {"dtype": "float32", "shape": [HAND_DIM]}
    features["teleop.right_hand_joints"] = {"dtype": "float32", "shape": [HAND_DIM]}
    features["states"] = {"dtype": "float32", "shape": [STATE_DIM]}
    features["action"] = {"dtype": "float32", "shape": [ACTION_DIM]}

    info = {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 1,
        "total_frames": int(n_frames),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": chunks_size,
        "fps": int(fps),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_index": 0,
                "task": instruction,
                "category": "default",
                "description": instruction,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_index": 0,
                "tasks": [0],
                "length": int(n_frames),
                "dataset_from_index": 0,
                "dataset_to_index": int(n_frames - 1),
                "robot_type": robot_type,
                "instruction": instruction,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

    if output_dir is not None:
        src_modality = output_dir / "meta" / "modality.json"
        if src_modality.is_file():
            shutil.copyfile(src_modality, meta_dir / "modality.json")


def write_episode_stats(out_dir: Path, actions: np.ndarray, fps: float) -> None:
    n = actions.shape[0]
    stats = {
        "episode_index": 0,
        "stats": {
            "action": {
                "min": actions.min(0).tolist(),
                "max": actions.max(0).tolist(),
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "count": [n],
            },
            "timestamp": {
                "min": [0.0],
                "max": [(n - 1) / fps],
                "mean": [((n - 1) / 2) / fps],
                "std": [n / (2 * fps * math.sqrt(3))],
                "count": [n],
            },
        },
    }
    path = out_dir / "meta" / "episodes_stats.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)
        f.write("\n")


def convert(
    pred_dir: Path,
    out_dir: Path,
    fps: int,
    instruction: str,
    output_dir: Path | None,
) -> Path:
    actions_path = pred_dir / "pred_actions.npy"
    if not actions_path.is_file():
        raise FileNotFoundError(f"pred_actions.npy not found: {actions_path}")

    actions = np.load(actions_path)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected pred_actions (T, {ACTION_DIM}), got {actions.shape}")

    states_path = pred_dir / "pred_states.npy"
    states_raw = np.load(states_path) if states_path.is_file() else None
    states = _pad_states(states_raw, actions.shape[0])

    template_info = None
    if output_dir is not None:
        info_path = output_dir / "meta" / "info.json"
        if info_path.is_file():
            template_info = json.loads(info_path.read_text(encoding="utf-8"))
            fps = int(template_info.get("fps", fps))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "data" / "chunk-000" / "episode_000000.parquet"
    write_episode_parquet(out_parquet, states, actions, fps=fps)
    write_meta(out_dir, actions.shape[0], instruction, fps, template_info, output_dir)
    write_episode_stats(out_dir, actions, fps=float(fps))

    np.savez_compressed(
        out_dir / "closeloop_actions.npz",
        pred_actions=actions,
        pred_states=states,
        fps=np.asarray([fps], dtype=np.int32),
    )
    print(f"[OK] actions {actions.shape} -> {out_parquet}")
    print(f"[OK] lerobot_v2.1 dataset -> {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert act_inference pred_actions.npy → lerobot_v2.1 for replay_real"
    )
    p.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing pred_actions.npy (and optional pred_states.npy)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output lerobot_v2.1 root (default: <pred-dir>/lerobot_v2.1)",
    )
    p.add_argument("--fps", type=int, default=FPS_DEFAULT)
    p.add_argument(
        "--instruction",
        type=str,
        default="closed-loop ACT pred replay",
        help="Task prompt written into meta/tasks.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional source lerobot_v2.1 (copy modality.json / fps / feature schema)",
    )
    args = p.parse_args()

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else pred_dir / "lerobot_v2.1"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    )

    convert(
        pred_dir=pred_dir,
        out_dir=out_dir,
        fps=args.fps,
        instruction=args.instruction,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
