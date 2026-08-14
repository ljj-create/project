import os
import sys
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import build_model, set_seed, load_config
from datasets.video_dataset import get_dataloader
from utils.metrics import (
    compute_map,
    make_predictions_dict,
    make_ground_truths_dict,
)


@torch.no_grad()
def test_model(model, dataloader, cfg, dataset_name, device):
    model.eval()
    model.clip_model.eval()

    if dataset_name == "ucf_crime":
        num_classes = cfg["datasets"]["ucf_crime"]["num_classes"]
    else:
        num_classes = cfg["datasets"]["xd_violence"]["num_classes"]

    iou_thresholds = cfg["evaluation"]["iou_thresholds"]
    all_predictions = {}
    all_ground_truths = {}
    global_idx = 0

    pbar = tqdm(dataloader, desc="Testing")
    for batch in pbar:
        frames = batch["frames"].to(device)
        class_labels = batch["class_idx"]
        num_frames = batch["num_frames"]

        outputs = model(frames, valid_lengths=num_frames)

        batch_size = class_labels.shape[0]
        local_preds = make_predictions_dict(outputs, num_classes,
                                            threshold=cfg["evaluation"]["inference_threshold"])
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

    logger.info("=" * 60)
    logger.info(f"Fine-Grained VAD Results on {dataset_name.upper()}")
    logger.info("=" * 60)
    header = "| " + " | ".join([f"IoU={t}" for t in iou_thresholds]) + " | AVG |"
    separator = "+" + "+".join(["-" * 10] * (len(iou_thresholds) + 1)) + "+"
    vals = "| " + " | ".join([f"{map_results[t]:>7.2f}%" for t in iou_thresholds]) + f" | {map_results['AVG']:>5.2f}% |"
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    logger.info(vals)
    logger.info(separator)

    logger.info("\n[SOTA Comparison - Paper Table 1 (best method: ExVAD)]")
    if dataset_name == "ucf_crime":
        sota_iou = [16.51, 12.35, 9.41, 7.82, 4.65]
        sota_avg = 10.15
        fine_vad_paper = [21.43, 18.62, 16.25, 11.41, 7.26]
        fine_vad_avg = 14.99
    else:
        sota_iou = [40.14, 32.75, 28.78, 20.15, 18.35]
        sota_avg = 28.23
        fine_vad_paper = [44.06, 36.69, 31.22, 25.80, 21.58]
        fine_vad_avg = 31.87

    our_iou = [map_results[t] for t in iou_thresholds]
    our_avg = map_results["AVG"]
    logger.info(f"  ExVAD (ICML'25 SOTA)       : IoU={sota_iou}  |  AVG={sota_avg:.2f}%")
    logger.info(f"  Fine-VAD (Paper Reported)  : IoU={fine_vad_paper}  |  AVG={fine_vad_avg:.2f}%")
    logger.info(f"  Our Reproduction           : IoU={[f'{v:.2f}' for v in our_iou]}  |  AVG={our_avg:.2f}%")
    if our_avg > sota_avg:
        gain = (our_avg - sota_avg) / sota_avg * 100
        logger.info(f"  *** Relative improvement over SOTA: +{gain:.1f}% (Paper reported: +47.7%)")

    return map_results


def main():
    parser = argparse.ArgumentParser("Test Fine-VAD")
    parser.add_argument("--dataset", type=str, default="ucf_crime",
                        choices=["ucf_crime", "xd_violence"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = load_config(args.config)
    model, _ = build_model(cfg, args.dataset, device)
    model.to(device)

    logger.info("Computing pseudo macro-categories (K={})...".format(cfg.progressive_learning.K))
    model.compute_pseudo_macro_categories()

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    test_loader, test_ds = get_dataloader(
        args.dataset, args.data_root, split="test",
        batch_size=cfg.training.batch_size,
        num_workers=0,
        interval=cfg.model.frame_sample_interval,
    )
    logger.info(f"Test samples: {len(test_ds)}")

    test_model(model, test_loader, cfg, args.dataset, device)


if __name__ == "__main__":
    main()
