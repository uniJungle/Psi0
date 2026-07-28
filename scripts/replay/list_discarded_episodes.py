#!/usr/bin/env python3
"""List discarded episodes for a SONIC/LeRobot dataset.

Usage examples:
  python scripts/replay/list_discarded_episodes.py \
    --data_dir /home/karthus_chen/ycb_ws/datasets/SONIC/xxx/data

  python scripts/replay/list_discarded_episodes.py \
    --dataset_dir /home/karthus_chen/ycb_ws/datasets/SONIC/xxx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolve_dataset_dir(data_dir: str | None, dataset_dir: str | None) -> Path:
    if data_dir:
        data_path = Path(data_dir).expanduser().resolve()
        if data_path.name == "data":
            return data_path.parent
        # Also support passing a chunk path like .../data/chunk-000
        if data_path.parent.name == "data":
            return data_path.parent.parent
        raise ValueError(f"--data_dir 应该指向 .../data 或 .../data/chunk-xxx，当前是: {data_path}")

    if dataset_dir:
        return Path(dataset_dir).expanduser().resolve()

    raise ValueError("必须提供 --data_dir 或 --dataset_dir")


def main() -> None:
    parser = argparse.ArgumentParser(description="查看数据集中哪些 episode 被标记为 discard")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录，例如 .../dataset/data")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集根目录，例如 .../dataset")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args()

    dataset_dir = _resolve_dataset_dir(args.data_dir, args.dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"未找到 info.json: {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    data_dir = dataset_dir / "data"
    total_episode_files = len(list(data_dir.glob("chunk-*/episode_*.parquet")))
    discarded = sorted(int(x) for x in info.get("discarded_episode_indices", []))
    data_tpl = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk_size = int(info.get("chunks_size", 1000))

    details = []
    for ep_idx in discarded:
        ep_chunk = ep_idx // chunk_size
        rel_path = data_tpl.format(episode_chunk=ep_chunk, episode_index=ep_idx)
        abs_path = dataset_dir / rel_path
        details.append(
            {
                "episode_index": ep_idx,
                "relative_path": rel_path,
                "exists": abs_path.is_file(),
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "total_episode_files": total_episode_files,
                    "discarded_count": len(discarded),
                    "discarded_episode_indices": discarded,
                    "files": details,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"总条数: {total_episode_files}")
    print(f"discard 数量: {len(discarded)}")
    print(f"discard index: {' '.join(str(i) for i in discarded) if discarded else '(none)'}")


if __name__ == "__main__":
    main()
