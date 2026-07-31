"""Shared pred-action I/O for ACT open-loop / closed-loop inference.

Layout::

    <data_root>/
      openloop_act_{ckpt_step}/episode_{id:06d}/   # open-loop
        pred_actions.npy
        meta/ data/ videos/ plots/
      closeloop_act_{ckpt_step}/                   # closed-loop
        pred_actions.npy
        meta/ data/ plots/
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

STATE_DIM = 33
ACTION_DIM = 68
TOKEN_DIM = 64
HAND_DIM = 2
FPS_DEFAULT = 30


def resolve_openloop_dir(data_root: Path, ckpt_step: int, episode_idx: int) -> Path:
    return Path(data_root) / f"openloop_act_{int(ckpt_step)}" / f"episode_{int(episode_idx):06d}"


def resolve_closeloop_dir(
    data_root: Path, ckpt_step: int, prefix: str = "act"
) -> Path:
    """Closed-loop output dir: ``closeloop_{prefix}_{ckpt_step}/``.

    ``prefix`` examples: ``act``, ``psi0``, ``gr00t_n1d7``.
    """
    return Path(data_root) / f"closeloop_{prefix}_{int(ckpt_step)}"


def save_pred_actions_npy(out_dir: Path, actions: np.ndarray) -> Path:
    """Save (T, 68) predictions as ``pred_actions.npy``."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected pred_actions (T, {ACTION_DIM}), got {actions.shape}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pred_actions.npy"
    np.save(path, actions)
    print(f"[SAVE] pred_actions {actions.shape} -> {path}")
    return path


def _pad_states(states: np.ndarray | None, n: int) -> np.ndarray:
    out = np.zeros((n, STATE_DIM), dtype=np.float32)
    if states is None:
        return out
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"states must be 2D, got {states.shape}")
    if states.shape[0] != n:
        raise ValueError(f"states T={states.shape[0]} != actions T={n}")
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
    states_list = [states[i].astype(np.float32) for i in range(n)]
    actions_list = [actions[i].astype(np.float32) for i in range(n)]
    token_list = [actions[i, :TOKEN_DIM].astype(np.float32) for i in range(n)]
    left_list = [
        actions[i, TOKEN_DIM : TOKEN_DIM + HAND_DIM].astype(np.float32) for i in range(n)
    ]
    right_list = [
        actions[i, TOKEN_DIM + HAND_DIM : TOKEN_DIM + 2 * HAND_DIM].astype(np.float32)
        for i in range(n)
    ]
    timestamps = np.asarray([float(i / fps) for i in range(n)], dtype=np.float64)
    frame_index = np.arange(n, dtype=np.int64)
    episode_index_arr = np.full(n, int(episode_index), dtype=np.int64)
    index_arr = np.arange(n, dtype=np.int64)
    task_index_arr = np.full(n, int(task_index), dtype=np.int64)
    next_done = np.zeros(n, dtype=bool)
    next_done[-1] = True

    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd

        rows = []
        for i in range(n):
            rows.append(
                {
                    "states": states_list[i].tolist(),
                    "action": actions_list[i].tolist(),
                    "action.motion_token": token_list[i].tolist(),
                    "teleop.left_hand_joints": left_list[i].tolist(),
                    "teleop.right_hand_joints": right_list[i].tolist(),
                    "timestamp": float(timestamps[i]),
                    "frame_index": int(frame_index[i]),
                    "episode_index": int(episode_index_arr[i]),
                    "index": int(index_arr[i]),
                    "task_index": int(task_index_arr[i]),
                    "next.done": bool(next_done[i]),
                }
            )
        pd.DataFrame(rows).to_parquet(out_parquet, index=False)
        return
    except ModuleNotFoundError:
        pass

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "states": states_list,
            "action": actions_list,
            "action.motion_token": token_list,
            "teleop.left_hand_joints": left_list,
            "teleop.right_hand_joints": right_list,
            "timestamp": timestamps,
            "frame_index": frame_index,
            "episode_index": episode_index_arr,
            "index": index_arr,
            "task_index": task_index_arr,
            "next.done": next_done,
        }
    )
    pq.write_table(table, out_parquet)


def write_meta(
    out_dir: Path,
    n_frames: int,
    instruction: str,
    fps: int,
    template_info: dict[str, Any] | None = None,
    template_dir: Path | None = None,
    keep_videos: bool = False,
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

    if not keep_videos:
        features = {
            k: v
            for k, v in features.items()
            if not (isinstance(v, dict) and v.get("dtype") == "video")
        }
        total_videos = 0
    else:
        total_videos = sum(
            1 for v in features.values() if isinstance(v, dict) and v.get("dtype") == "video"
        )

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
        "total_videos": int(total_videos),
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

    if template_dir is not None:
        src_modality = Path(template_dir) / "meta" / "modality.json"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)
        f.write("\n")


def write_pred_lerobot(
    out_dir: Path,
    actions: np.ndarray,
    states: np.ndarray | None = None,
    fps: int = FPS_DEFAULT,
    instruction: str = "ACT pred replay",
    template_dir: Path | None = None,
    keep_videos: bool = False,
) -> Path:
    """Write a single-episode lerobot_v2.1 dataset for ``replay_real.py --mode token``."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions (T, {ACTION_DIM}), got {actions.shape}")

    states33 = _pad_states(states, actions.shape[0])
    template_info = None
    template_dir = Path(template_dir).resolve() if template_dir is not None else None
    if template_dir is not None:
        info_path = template_dir / "meta" / "info.json"
        if info_path.is_file():
            template_info = json.loads(info_path.read_text(encoding="utf-8"))
            fps = int(template_info.get("fps", fps))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "data" / "chunk-000" / "episode_000000.parquet"
    write_episode_parquet(out_parquet, states33, actions, fps=float(fps))
    write_meta(
        out_dir,
        n_frames=actions.shape[0],
        instruction=instruction,
        fps=int(fps),
        template_info=template_info,
        template_dir=template_dir,
        keep_videos=keep_videos,
    )
    write_episode_stats(out_dir, actions, fps=float(fps))
    print(f"[SAVE] lerobot_v2.1 parquet -> {out_parquet}")
    return out_dir
