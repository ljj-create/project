"""CLIP frame-feature cache.

Since the CLIP image encoder is frozen (``freeze_clip: true``), the frame-level
CLIP features never change across epochs. This module pre-computes them once,
stores them on disk, and serves them back to training/evaluation so that each
epoch only runs the trainable LGT-Adapter instead of re-encoding every frame.

Usage (from train.py / test.py):
    python train.py --cache-dir ./clip_cache        # precompute + train on cache
    python test.py  --cache-dir ./clip_cache --checkpoint <ckpt>
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.video_dataset import collate_fn_batch


class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-computed frozen CLIP frame features instead of videos."""

    def __init__(self, source_dataset, cache_dir):
        self.source = source_dataset
        self.cache_dir = cache_dir
        # Reuse the source metadata (class / binary labels / names).
        self.samples = source_dataset.samples

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.cache_dir, f"{idx}.pt"), map_location="cpu")
        sample = self.samples[idx]
        return {
            "features": data["features"],          # (N, D) float32
            "num_frames": int(data["num_frames"]),
            "class_idx": sample["class_idx"],
            "binary_label": sample["binary_label"],
            "class_name": sample["class_name"],
        }


def collate_features(batch):
    """Collate cached feature dicts, zero-padding the variable frame axis."""
    feats_list = [item["features"] for item in batch]
    max_len = max(f.shape[0] for f in feats_list)
    D = feats_list[0].shape[-1]
    padded = []
    for f in feats_list:
        if f.shape[0] < max_len:
            pad = torch.zeros(max_len - f.shape[0], D, dtype=f.dtype)
            padded.append(torch.cat([f, pad], dim=0))
        else:
            padded.append(f)
    return {
        "features": torch.stack(padded, dim=0),
        "num_frames": torch.tensor([item["num_frames"] for item in batch], dtype=torch.long),
        "class_idx": torch.tensor([item["class_idx"] for item in batch], dtype=torch.long),
        "binary_label": torch.tensor([item["binary_label"] for item in batch], dtype=torch.long),
        "class_names": [item["class_name"] for item in batch],
    }


@torch.no_grad()
def precompute_clip_features(model, dataset, cache_dir, device, batch_size=16, logger=None):
    """Encode every video once with the frozen CLIP encoder and cache to ``cache_dir``.

    Cache files are named ``{sample_idx}.pt`` with ``{"features": (N, D), "num_frames": N}``.
    Already-cached samples are skipped, so the run can be resumed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    total = len(dataset)
    done = sum(1 for f in os.listdir(cache_dir) if f.endswith(".pt"))
    if done >= total:
        msg = f"[feature-cache] {cache_dir}: already complete ({done}/{total})"
        print(msg) if logger is None else logger.info(msg)
        return

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn_batch,
    )
    cached = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        num_frames = batch["num_frames"]
        B, N, C, H, W = frames.shape

        # Only the frozen CLIP encoder runs here; adapter + fusion stay out.
        if (num_frames == N).all():
            valid_mask = None
        else:
            lengths = num_frames.to(device)
            positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            valid_mask = positions < lengths.unsqueeze(1)
        feats = model.clip_encode_frames(frames, valid_mask)   # (B, N, D)

        for i in range(B):
            idx = cached + i
            cache_file = os.path.join(cache_dir, f"{idx}.pt")
            if os.path.exists(cache_file):
                continue
            nf = int(num_frames[i])
            torch.save({
                "features": feats[i, :nf].float().cpu(),
                "num_frames": nf,
            }, cache_file)
        cached += B
        msg = f"[feature-cache] {cache_dir}: {min(cached, total)}/{total}"
        print(msg) if logger is None else logger.info(msg)

    msg = f"[feature-cache] {cache_dir}: complete ({total}/{total})"
    print(msg) if logger is None else logger.info(msg)


def get_feature_dataloader(model, dataset_name, data_root, split, cache_dir,
                           batch_size, interval, device, logger=None):
    """Build the source dataset, precompute/load the CLIP cache, return the feature loader."""
    from datasets.video_dataset import get_dataloader

    _, dataset = get_dataloader(
        dataset_name, data_root, split,
        batch_size=1, num_workers=0, interval=interval,
    )
    precompute_clip_features(model, dataset, cache_dir, device, logger=logger)

    cached_ds = CachedFeatureDataset(dataset, cache_dir)
    loader = DataLoader(
        cached_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        collate_fn=collate_features,
        drop_last=(split == "train"),
    )
    return loader, cached_ds
