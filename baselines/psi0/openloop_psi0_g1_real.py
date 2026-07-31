#!/usr/bin/env python3
"""Open-loop Psi0 inference on a SONIC lerobot_v2.1 episode.

Rollout (same cadence as ACT open-loop):

  1. At frame ``t``, query policy with GT image[t] + state[t].
  2. Write ``chunk[0 : n_action_steps]`` to frames ``t .. t+n_action_steps-1``.
  3. Advance ``t += n_action_steps`` and repeat.

Output layout::

    <data_root>/openloop_psi0_{ckpt_step}/episode_{id:06d}/
      pred_actions.npy
      meta/ data/ videos/ plots/

Requires a running Psi0 HTTP server (`/act` + `/health`), e.g.:

    bash scripts/deploy/serve_psi0_simple.sh <RUN_DIR> <CKPT_STEP>
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from base64 import b64decode, b64encode
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from tqdm.auto import tqdm

_PSI0_ROOT = Path(__file__).resolve().parents[2]
if str(_PSI0_ROOT) not in sys.path:
    sys.path.insert(0, str(_PSI0_ROOT))

from baselines.act.pred_action_io import save_pred_actions_npy

IMAGE_KEY_DEFAULT = "observation.images.egocentric_right"
STATE_DIM = 33
ACTION_DIM = 68
TOKEN_DIM = 64
HAND_DIM = 2
FPS_DEFAULT = 30


def resolve_openloop_dir(data_root: Path, ckpt_step: int, episode_idx: int) -> Path:
    return Path(data_root) / f"openloop_psi0_{int(ckpt_step)}" / f"episode_{int(episode_idx):06d}"


def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(dct["shape"]) if dct["shape"] else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


def load_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_episode_parquet(dataset_dir: Path, episode_idx: int, info: dict) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = episode_idx // chunks_size
    data_tpl = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    path = dataset_dir / data_tpl.format(episode_chunk=chunk_idx, episode_index=episode_idx)
    if not path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return path


def resolve_episode_video(
    dataset_dir: Path, episode_idx: int, info: dict, video_key: str
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = episode_idx // chunks_size
    video_tpl = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    path = dataset_dir / video_tpl.format(
        episode_chunk=chunk_idx, episode_index=episode_idx, video_key=video_key
    )
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def resolve_task_prompt(dataset_dir: Path, episode_idx: int) -> str:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                if int(ep.get("episode_index", -1)) != episode_idx:
                    continue
                if ep.get("instruction"):
                    return str(ep["instruction"])
                tasks = ep.get("tasks") or []
                if isinstance(tasks, list) and tasks and isinstance(tasks[0], str):
                    return str(tasks[0])
                break
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("task"):
                    return str(item["task"])
                break
    return "walk to table and place apple on pink plate"


def list_video_keys(info: dict) -> list[str]:
    keys = []
    for k, v in info.get("features", {}).items():
        if isinstance(v, dict) and v.get("dtype") == "video":
            keys.append(k)
    return keys


class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._idx = -1

    def read_rgb(self, frame_idx: int) -> np.ndarray:
        if frame_idx != self._idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.path}")
        self._idx = frame_idx
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class PSIHTTPClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 22085, timeout: float = 30.0):
        self.url = f"http://{host}:{port}/act"
        self.health_url = f"http://{host}:{port}/health"
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> bool:
        try:
            resp = self.session.get(self.health_url, timeout=self.timeout)
            return resp.ok and resp.json().get("status") == "ok"
        except Exception as e:
            print(f"[PSI0] health check failed: {e}")
            return False

    def query_action(
        self,
        image: np.ndarray,
        states: np.ndarray,
        instruction: str,
        image_key: str,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32).reshape(1, -1)
        payload = {
            "image": {image_key: np.asarray(image, dtype=np.uint8)},
            "state": {"states": states},
            "instruction": instruction,
            "history": {},
            "condition": {},
            "gt_action": [],
            "dataset_name": "openloop",
            "timestamp": str(datetime.now()).replace(" ", "_").replace(":", "-"),
        }
        resp = self.session.post(
            self.url,
            json=convert_numpy_in_dict(payload, numpy_serialize),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = convert_numpy_in_dict(resp.json(), numpy_deserialize)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                raise RuntimeError(f"Psi0 server returned non-JSON string: {data}")
        if not isinstance(data, dict):
            raise RuntimeError(f"Psi0 server returned unexpected response type: {type(data)}")
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        if "status" in data and "action" not in data:
            raise RuntimeError(f"Psi0 server status error: {data['status']}")
        action = np.asarray(data["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        return action


def write_episode_parquet(
    out_parquet: Path,
    states: np.ndarray,
    actions_pred: np.ndarray,
    fps: float,
    episode_index: int = 0,
    task_index: int = 0,
) -> None:
    n = states.shape[0]
    rows = []
    for i in range(n):
        act = actions_pred[i].astype(np.float32)
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
    src_info: dict,
    src_dataset_dir: Path,
    n_frames: int,
    instruction: str,
    fps: int,
) -> None:
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    video_keys = list_video_keys(src_info)
    features = dict(src_info.get("features", {}))
    features["action.motion_token"] = {"dtype": "float32", "shape": [TOKEN_DIM]}
    features["teleop.left_hand_joints"] = {"dtype": "float32", "shape": [HAND_DIM]}
    features["teleop.right_hand_joints"] = {"dtype": "float32", "shape": [HAND_DIM]}

    info = {
        "codebase_version": src_info.get("codebase_version", "v2.1"),
        "robot_type": src_info.get("robot_type", "g1"),
        "total_episodes": 1,
        "total_frames": int(n_frames),
        "total_tasks": 1,
        "total_videos": len(video_keys),
        "total_chunks": 1,
        "chunks_size": int(src_info.get("chunks_size", 1000)),
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
                "robot_type": "g1",
                "instruction": instruction,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

    src_modality = src_dataset_dir / "meta" / "modality.json"
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


def copy_videos(
    src_dataset_dir: Path,
    src_episode_idx: int,
    src_info: dict,
    out_dir: Path,
    out_episode_idx: int = 0,
) -> None:
    video_keys = list_video_keys(src_info)
    chunks_size = int(src_info.get("chunks_size", 1000))
    out_chunk = out_episode_idx // chunks_size
    for key in video_keys:
        src = resolve_episode_video(src_dataset_dir, src_episode_idx, src_info, key)
        dst = (
            out_dir
            / "videos"
            / f"chunk-{out_chunk:03d}"
            / key
            / f"episode_{out_episode_idx:06d}.mp4"
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[IO] copied video {key} -> {dst}")


HAND_DIM_NAMES = ("L_thumb_aux", "L_others", "R_thumb_aux", "R_others")


def _dim_label(dim: int) -> str:
    if dim < TOKEN_DIM:
        return f"token[{dim}]"
    hand_i = dim - TOKEN_DIM
    name = HAND_DIM_NAMES[hand_i] if hand_i < len(HAND_DIM_NAMES) else f"hand[{hand_i}]"
    return f"action[{dim}] {name}"


def plot_pred_vs_gt(
    gt: np.ndarray,
    pred: np.ndarray,
    plots_dir: Path,
    episode_idx: int,
) -> None:
    assert gt.shape == pred.shape
    assert gt.shape[1] == ACTION_DIM
    t = np.arange(gt.shape[0])
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        ACTION_DIM,
        1,
        figsize=(14, 1.1 * ACTION_DIM),
        sharex=True,
        constrained_layout=True,
    )
    if ACTION_DIM == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(t, gt[:, dim], linestyle="--", alpha=0.75, color="C1", label="gt", linewidth=1.0)
        ax.plot(t, pred[:, dim], linestyle="-", alpha=0.95, color="C0", label="pred", linewidth=1.0)
        ax.set_ylabel(_dim_label(dim), fontsize=7, rotation=0, labelpad=55, va="center")
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title(f"Psi0 open-loop ep{episode_idx}  |  all {ACTION_DIM} dims", fontsize=11)

    axes[-1].set_xlabel("frame index", fontsize=9)
    out_path = plots_dir / f"openloop_pred_vs_gt_all_dims_eps{episode_idx}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Psi0 open-loop inference → lerobot_v2.1")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Task dataset root, e.g. .../SONIC/walk_to_table_and_place_apple_on_pink_plate. "
            "Writes to <data-root>/openloop_psi0_{ckpt_step}/episode_{id:06d}/"
        ),
    )
    p.add_argument(
        "--ckpt-step",
        type=int,
        required=True,
        help="Checkpoint step used by the Psi0 server (used in output subdir name)",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Source lerobot_v2.1 root (default: <data-root>/lerobot_v2.1)",
    )
    p.add_argument("--episode-idx", type=int, required=True, help="Source episode index")
    p.add_argument("--host", type=str, default="localhost", help="Psi0 policy server host")
    p.add_argument("--port", type=int, default=22085, help="Psi0 policy server port")
    p.add_argument(
        "--image-key",
        type=str,
        default=IMAGE_KEY_DEFAULT,
        help="Dataset / policy image key",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=1,
        help="Frames to write per query: chunk[0:k] at t..t+k-1, then re-query at t+k. "
        "1 = every frame uses chunk[0] only.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on frames (0 = full episode)",
    )
    p.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override task instruction (default: from meta/episodes.jsonl)",
    )
    return p.parse_args()


def rollout_pred_actions(
    policy: PSIHTTPClient,
    video: VideoReader,
    states: np.ndarray,
    n_frames: int,
    n_action_steps: int,
    instruction: str,
    image_key: str,
) -> tuple[np.ndarray, list[int]]:
    pred = np.zeros((n_frames, ACTION_DIM), dtype=np.float32)
    written = np.zeros(n_frames, dtype=bool)
    query_frames: list[int] = []
    n_action_steps = max(1, n_action_steps)

    t = 0
    pbar = tqdm(total=n_frames, desc="openloop rollout", unit="frame")
    while t < n_frames:
        image = video.read_rgb(t)
        chunk = policy.query_action(
            image=image,
            states=states[t],
            instruction=instruction,
            image_key=image_key,
        )
        if chunk.ndim != 2 or chunk.shape[-1] != ACTION_DIM:
            raise RuntimeError(f"bad action shape {chunk.shape}, expected (T, {ACTION_DIM})")

        steps = min(n_action_steps, chunk.shape[0], n_frames - t)
        for k in range(steps):
            pred[t + k] = chunk[k].astype(np.float32)
            written[t + k] = True
        query_frames.append(t)
        pbar.update(steps)
        t += steps
    pbar.close()

    if not written.all():
        last = None
        for i in range(n_frames - 1, -1, -1):
            if written[i]:
                last = pred[i].copy()
                break
        if last is not None:
            for i in range(n_frames):
                if not written[i]:
                    pred[i] = last

    return pred, query_frames


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else (data_root / "lerobot_v2.1").resolve()
    )
    out_dir = resolve_openloop_dir(data_root, args.ckpt_step, args.episode_idx)
    info = load_info(dataset_dir)
    fps = int(info.get("fps", FPS_DEFAULT))
    print(f"[IO] data_root={data_root}")
    print(f"[IO] dataset_dir={dataset_dir}")
    print(f"[IO] output episode dir={out_dir}")

    parquet_path = resolve_episode_parquet(dataset_dir, args.episode_idx, info)
    print(f"[DATA] parquet={parquet_path}")
    df = pd.read_parquet(parquet_path)
    n = len(df)
    if args.max_frames > 0:
        n = min(n, args.max_frames)
        df = df.iloc[:n].reset_index(drop=True)

    states = np.vstack([np.asarray(x, dtype=np.float32) for x in df["states"]])
    gt_actions = np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"]])
    if states.shape[1] != STATE_DIM:
        raise ValueError(f"states dim {states.shape[1]} != {STATE_DIM}")
    if gt_actions.shape[1] != ACTION_DIM:
        raise ValueError(f"action dim {gt_actions.shape[1]} != {ACTION_DIM}")

    instruction = args.instruction or resolve_task_prompt(dataset_dir, args.episode_idx)
    print(f"[DATA] episode={args.episode_idx} frames={n} instruction={instruction!r}")

    video_path = resolve_episode_video(dataset_dir, args.episode_idx, info, args.image_key)
    video = VideoReader(video_path)
    print(f"[DATA] image video={video_path}")

    policy = PSIHTTPClient(host=args.host, port=args.port)
    print(f"[PSI0] Checking server {args.host}:{args.port} ...")
    if not policy.health_check():
        print("[PSI0] server not healthy. Start scripts/deploy/serve_psi0_simple.sh first.")
        sys.exit(1)
    print(
        f"[ROLLOUT] n_action_steps={max(1, args.n_action_steps)} "
        f"(server must return chunk length >= this)"
    )

    try:
        pred_actions, query_frames = rollout_pred_actions(
            policy=policy,
            video=video,
            states=states,
            n_frames=n,
            n_action_steps=args.n_action_steps,
            instruction=instruction,
            image_key=args.image_key,
        )
    finally:
        video.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pred_actions_npy(out_dir, pred_actions)

    out_parquet = out_dir / "data" / "chunk-000" / "episode_000000.parquet"
    write_episode_parquet(out_parquet, states, pred_actions, fps=fps, episode_index=0)
    print(f"[IO] wrote {out_parquet}")

    copy_videos(dataset_dir, args.episode_idx, info, out_dir, out_episode_idx=0)
    write_meta(out_dir, info, dataset_dir, n_frames=n, instruction=instruction, fps=fps)
    write_episode_stats(out_dir, pred_actions, fps=fps)

    np.save(out_dir / "gt_actions.npy", gt_actions.astype(np.float32))
    np.save(out_dir / "query_frames.npy", np.asarray(query_frames, dtype=np.int64))

    plot_pred_vs_gt(gt_actions, pred_actions, out_dir / "plots", args.episode_idx)

    print("\n=== Open-loop summary ===")
    print(f"frames={n}  n_action_steps={max(1, args.n_action_steps)}  queries={len(query_frames)}")
    print(f"query_frames (first 10): {query_frames[:10]}{'...' if len(query_frames) > 10 else ''}")
    print(f"ckpt_step={args.ckpt_step}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
