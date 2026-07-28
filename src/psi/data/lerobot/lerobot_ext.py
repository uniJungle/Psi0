from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import os
import packaging.version
from pathlib import Path

import torch
from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform


def _patch_lerobot_local_only(local_root: Path) -> None:
    """Use local LeRobot files only; do not fall back to huggingface.co."""
    if not (local_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Local LeRobot dataset not found: {local_root} (missing meta/info.json)"
        )

    import huggingface_hub
    import lerobot.datasets.lerobot_dataset as lrd
    import lerobot.datasets.utils as lru

    if getattr(lrd, "_psi_local_only_patch", False):
        return

    _orig_snapshot_download = huggingface_hub.snapshot_download

    def _local_get_safe_version(repo_id: str, version):
        v = (
            packaging.version.parse(version)
            if not isinstance(version, packaging.version.Version)
            else version
        )
        return f"v{v}"

    def _local_snapshot_download(repo_id, *args, **kwargs):
        local_dir = kwargs.get("local_dir")
        if local_dir is not None and Path(local_dir, "meta", "info.json").is_file():
            return str(local_dir)
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError(
                f"HF_HUB_OFFLINE=1 and local dataset incomplete at {local_dir}"
            )
        return _orig_snapshot_download(repo_id, *args, **kwargs)

    lru.get_safe_version = _local_get_safe_version
    lrd.get_safe_version = _local_get_safe_version
    lrd.snapshot_download = _local_snapshot_download
    lrd._psi_local_only_patch = True


class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
        self, 
        data_cfg: LerobotDataConfig, 
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        local_root = Path(resolve_path(f"{data_cfg.root_dir}/{first_repo}"))
        _patch_lerobot_local_only(local_root)

        dataset_meta = LeRobotDatasetMetadata(first_repo, local_root)
        assert isinstance(data_cfg.transform.repack, LerobotRepackTransform)
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)

        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
        else:
            repo_ids = first_repo
            root_dir = local_root
            lerobot_dataset_class = LeRobotDataset

        self.base_dataset = lerobot_dataset_class(
            repo_ids,# type: ignore
            root=root_dir,
            delta_timestamps=delta_timestamps, # type: ignore
            image_transforms=None,
            download_videos=False,
        )
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        return self.base_dataset[idx]
    
    def __len__(self):
        return len(self.base_dataset)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property
    def num_frames(self):
        return self.base_dataset.num_frames
    
    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore
