import os
import sys
import time
import yaml
import random
import argparse
import numpy as np
import torch
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import open_clip
except ImportError:
    logger.error("open_clip not installed. Please run: pip install open-clip-torch")
    raise

from models.fine_vad import FineVAD, CoarseContrastiveLoss
from datasets.video_dataset import get_dataloader, UCF_CRIME_CLASSES, XD_VIOLENCE_CLASSES
from utils.metrics import (
    binary_cross_entropy_loss,
    weighted_cross_entropy_loss,
    compute_map,
    make_predictions_dict,
    make_ground_truths_dict,
)
from utils.feature_cache import get_feature_dataloader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path) -> Any:
    """Load YAML config as an attribute-accessible dict (EasyDict).

    Returns Any on purpose: EasyDict exposes dynamic attributes that a
    static type checker cannot resolve, and callers access config as
    ``cfg.training.batch_size`` and/or ``cfg["training"]["batch_size"]``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from easydict import EasyDict
    return EasyDict(cfg)


def build_model(cfg, dataset_name, device, allow_random_init=False):
    logger.info(f"Loading CLIP backbone: {cfg.model.clip_arch} ({cfg.model.clip_pretrained})")
    try:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model.clip_arch,
            pretrained=cfg.model.clip_pretrained,
            device=device,
        )
        logger.info("CLIP pretrained weights loaded successfully.")
    except Exception as e:
        if allow_random_init:
            logger.warning(f"Failed to load CLIP pretrained weights ({e}). "
                           f"Falling back to RANDOM INITIALIZATION for verification only.")
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.model.clip_arch,
                pretrained="",
                device=device,
            )
        else:
            raise
    tokenizer = open_clip.get_tokenizer(cfg.model.clip_arch)

    if cfg.model.freeze_clip:
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()

    if dataset_name == "ucf_crime":
        anomaly_classes = cfg.datasets.ucf_crime.anomaly_classes
    else:
        anomaly_classes = cfg.datasets.xd_violence.anomaly_classes

    num_anomaly_classes = len(anomaly_classes)

    model = FineVAD(
        clip_model=clip_model,
        clip_tokenizer=tokenizer,
        num_anomaly_classes=num_anomaly_classes,
        anomaly_class_names=anomaly_classes,
        text_prompts=cfg.text_prompts,
        dataset_name=dataset_name,
        K=cfg.progressive_learning.K,
        T=cfg.progressive_learning.T,
        feature_dim=cfg.model.feature_dim,
        omega=cfg.progressive_learning.omega,
        device=device,
    )
    return model, preprocess


def train_one_epoch(model, dataloader, optimizer, contrastive_loss_fn,
                    cfg, dataset_name, epoch, device, use_cache=False):
    model.train()
    model.clip_model.eval()

    if dataset_name == "ucf_crime":
        train_cfg = cfg.training.ucf_crime
        num_classes = cfg.datasets.ucf_crime.num_classes
    else:
        train_cfg = cfg.training.xd_violence
        num_classes = cfg.datasets.xd_violence.num_classes

    lambda1 = train_cfg.lambda1
    lambda2 = train_cfg.lambda2
    omega = cfg.progressive_learning.omega

    total_loss = 0.0
    total_bce = 0.0
    total_cts = 0.0
    total_refine = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
    for batch in pbar:
        frames = batch["frames"].to(device)
        class_labels = batch["class_idx"].to(device)
        binary_labels = batch["binary_label"].to(device)
        num_frames = batch["num_frames"]

        optimizer.zero_grad()

        outputs = model(frames, valid_lengths=num_frames)
        S_coarse = outputs["S_coarse"]
        S_fine = outputs["S_fine"]
        f_video = outputs["f_video"]

        loss_bce = binary_cross_entropy_loss(S_coarse, binary_labels)
        loss_cts = contrastive_loss_fn(f_video, binary_labels)
        macro_priors = getattr(model, "macro_category_priors", None)
        loss_refine = weighted_cross_entropy_loss(
            S_fine, class_labels, macro_priors, num_classes, omega
        )

        loss = loss_bce + lambda1 * loss_cts + lambda2 * loss_refine

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_bce += loss_bce.item()
        total_cts += loss_cts.item()
        total_refine += loss_refine.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "bce": f"{loss_bce.item():.4f}",
            "cts": f"{loss_cts.item():.4f}",
            "refine": f"{loss_refine.item():.4f}",
        })

    return {
        "loss": total_loss / max(n_batches, 1),
        "bce": total_bce / max(n_batches, 1),
        "cts": total_cts / max(n_batches, 1),
        "refine": total_refine / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, cfg, dataset_name, device):
    model.eval()
    model.clip_model.eval()

    if dataset_name == "ucf_crime":
        num_classes = cfg.datasets.ucf_crime.num_classes
    else:
        num_classes = cfg.datasets.xd_violence.num_classes

    iou_thresholds = cfg.evaluation.iou_thresholds
    all_predictions = {}
    all_ground_truths = {}
    global_idx = 0

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        frames = batch["frames"].to(device)
        class_labels = batch["class_idx"]
        num_frames = batch["num_frames"]

        outputs = model(frames, valid_lengths=num_frames)

        batch_size = class_labels.shape[0]
        local_preds = make_predictions_dict(outputs, num_classes,
                                            threshold=cfg.evaluation.inference_threshold)
        local_gts = make_ground_truths_dict(class_labels, num_frames, num_classes)

        # mAP is computed only for abnormal videos (paper Sec. 4.1).
        cls_arr = class_labels.cpu().numpy() if hasattr(class_labels, "cpu") else np.asarray(class_labels)
        for b in range(batch_size):
            if cls_arr[b] == 0:
                continue
            pk, gk = f"video_{b}", f"video_{b}"
            all_predictions[f"g{global_idx}_{pk}"] = local_preds.get(pk, {})
            all_ground_truths[f"g{global_idx}_{gk}"] = local_gts.get(gk, {})
        global_idx += 1

    map_results = compute_map(
        all_predictions, all_ground_truths,
        iou_thresholds=iou_thresholds,
        num_classes=num_classes,
    )
    return map_results


def main():
    parser = argparse.ArgumentParser("Fine-VAD Reproduction (Paper Implementation)")
    parser.add_argument("--dataset", type=str, default="ucf_crime",
                        choices=["ucf_crime", "xd_violence"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_config(args.config)
    cfg.training.device = str(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model, _ = build_model(cfg, args.dataset, device)
    model.to(device)

    logger.info("Computing pseudo macro-categories via K-Means (K={})...".format(cfg.progressive_learning.K))
    model.compute_pseudo_macro_categories()
    logger.info(f"Done. Pseudo-labels: {model.pseudo_macro_labels}")

    contrastive_loss_fn = CoarseContrastiveLoss(temperature=0.07)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    train_loader, train_ds = get_dataloader(
        args.dataset, args.data_root, split="train",
        batch_size=cfg.training.batch_size,
        num_workers=0,
        interval=cfg.model.frame_sample_interval,
    )
    test_loader, test_ds = get_dataloader(
        args.dataset, args.data_root, split="test",
        batch_size=cfg.training.batch_size,
        num_workers=0,
        interval=cfg.model.frame_sample_interval,
    )
    logger.info(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from epoch {start_epoch}")

    if args.eval_only:
        logger.info("Running evaluation only...")
        results = evaluate(model, test_loader, cfg, args.dataset, device)
        for k, v in results.items():
            logger.info(f"  mAP@{k:<4} = {v:.2f}%")
        return

    best_avg_map = -1.0
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"best_{args.dataset}.pt")

    logger.info("=" * 80)
    logger.info(f"Starting training on {args.dataset.upper()}")
    logger.info(f"  CLIP: {cfg.model.clip_arch} ({cfg.model.clip_pretrained})")
    logger.info(f"  K={cfg.progressive_learning.K}  T={cfg.progressive_learning.T}  omega={cfg.progressive_learning.omega}")
    if args.dataset == "ucf_crime":
        logger.info(f"  lambda1={cfg.training.ucf_crime.lambda1}  lambda2={cfg.training.ucf_crime.lambda2}")
    else:
        logger.info(f"  lambda1={cfg.training.xd_violence.lambda1}  lambda2={cfg.training.xd_violence.lambda2}")
    logger.info(f"  Batch={cfg.training.batch_size}  LR={cfg.training.learning_rate}  Epochs={cfg.training.num_epochs}")
    logger.info("=" * 80)

    eval_interval = cfg.training.get("eval_interval", 1)

    for epoch in range(start_epoch, cfg.training.num_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, contrastive_loss_fn,
            cfg, args.dataset, epoch, device,
        )
        logger.info(f"[Epoch {epoch + 1}] Train - loss={train_metrics['loss']:.4f}  "
                    f"bce={train_metrics['bce']:.4f}  cts={train_metrics['cts']:.4f}  "
                    f"refine={train_metrics['refine']:.4f}")

        if (epoch + 1) % eval_interval == 0 or epoch == cfg.training.num_epochs - 1:
            results = evaluate(model, test_loader, cfg, args.dataset, device)
            logger.info(f"[Epoch {epoch + 1}] Test mAP results:")
            for k, v in results.items():
                logger.info(f"  mAP@{k:<4} = {v:.2f}%")

            avg_map = results.get("AVG", 0.0)
            if avg_map > best_avg_map:
                best_avg_map = avg_map
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results": results,
                    "pseudo_macro_labels": model.pseudo_macro_labels,
                    "config": cfg,
                }, best_ckpt_path)
                logger.info(f"*** Saved best checkpoint (AVG mAP={avg_map:.2f}%) to {best_ckpt_path}")

            last_ckpt_path = os.path.join(args.checkpoint_dir, f"last_{args.dataset}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "results": results,
            }, last_ckpt_path)

    logger.info("=" * 80)
    logger.info(f"Training complete. Best AVG mAP = {best_avg_map:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
