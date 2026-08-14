import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F


def weighted_cross_entropy_loss(s_fine, class_labels, macro_priors, num_classes, omega=2.0):
    """Weighted cross-entropy of Eq. (9):  alpha_ij = omega if j in C_p(i), else 1."""
    B = s_fine.shape[0]
    log_probs = F.log_softmax(s_fine, dim=-1)

    if isinstance(class_labels, torch.Tensor):
        class_labels_t = class_labels
    else:
        class_labels_t = torch.from_numpy(np.asarray(class_labels)).long().to(s_fine.device)

    one_hot = torch.zeros_like(log_probs)
    one_hot.scatter_(1, class_labels_t.unsqueeze(1), 1.0)
    weights = torch.ones(B, num_classes, device=s_fine.device)

    for i in range(B):
        cls_idx = int(class_labels_t[i])
        if cls_idx == 0:
            continue
        anomaly_idx = cls_idx - 1
        if macro_priors is not None and anomaly_idx in macro_priors:
            pseudo_macro = macro_priors[anomaly_idx]
            for j in range(1, num_classes):
                a_j = j - 1
                if a_j in macro_priors and macro_priors[a_j] == pseudo_macro:
                    weights[i, j] = omega

    weighted_log_probs = weights * one_hot * log_probs
    loss = -weighted_log_probs.sum(dim=-1).mean()
    return loss


def binary_cross_entropy_loss(s_coarse, binary_labels):
    """Coarse-level BCE of Eq. (8) on the top-T-pooled anomaly score s_coarse[:, 1]."""
    s_anom = s_coarse[:, 1]
    # Cosine similarity lives in [-1, 1]; clamp keeps log() finite.
    s_anom_clamped = torch.clamp(s_anom, min=1e-7, max=1.0 - 1e-7)

    if isinstance(binary_labels, torch.Tensor):
        labels = binary_labels.float()
    else:
        labels = torch.from_numpy(np.asarray(binary_labels)).float().to(s_coarse.device)

    loss = -(labels * torch.log(s_anom_clamped) +
             (1.0 - labels) * torch.log(1.0 - s_anom_clamped))
    return loss.mean()


def compute_map(predictions, ground_truths, iou_thresholds, num_classes: int):
    results = {}
    for iou_t in iou_thresholds:
        avg_precisions = []
        for c in range(1, num_classes):
            ap = compute_class_ap(predictions, ground_truths, c, iou_t)
            if ap is not None:
                avg_precisions.append(ap)
        if avg_precisions:
            results[iou_t] = float(np.mean(avg_precisions) * 100)
        else:
            results[iou_t] = 0.0
    results["AVG"] = float(np.mean(list(results.values()))) if results else 0.0
    return results


def compute_class_ap(predictions, ground_truths, class_idx, iou_threshold):
    detection_list = []
    gt_by_video = defaultdict(list)

    for video_id, pred in predictions.items():
        for det in pred.get(class_idx, []):
            detection_list.append({
                "video_id": video_id,
                "confidence": det["confidence"],
                "start": det["start"],
                "end": det["end"],
                "matched": False,
            })

    for video_id, gt in ground_truths.items():
        for g in gt.get(class_idx, []):
            gt_by_video[video_id].append({
                "start": g["start"],
                "end": g["end"],
                "matched": False,
            })

    detection_list.sort(key=lambda x: x["confidence"], reverse=True)

    tp = np.zeros(len(detection_list))
    fp = np.zeros(len(detection_list))
    total_gt = sum(len(v) for v in gt_by_video.values())
    if total_gt == 0:
        return None

    for i, det in enumerate(detection_list):
        video_gts = gt_by_video.get(det["video_id"], [])
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(video_gts):
            if gt["matched"]:
                continue
            iou = segment_iou(det["start"], det["end"], gt["start"], gt["end"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            video_gts[best_gt_idx]["matched"] = True
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(1, mrec.size):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def segment_iou(s1, e1, s2, e2):
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0
    return intersection / union


def extract_detections_from_scores(
    frame_scores,
    class_idx,
    threshold=0.5,
    min_segment_length=2,
):
    detections = []
    in_segment = False
    seg_start = 0
    seg_conf_sum = 0.0
    seg_count = 0

    for t, score in enumerate(frame_scores):
        if score >= threshold and not in_segment:
            in_segment = True
            seg_start = t
            seg_conf_sum = float(score)
            seg_count = 1
        elif score >= threshold and in_segment:
            seg_conf_sum += float(score)
            seg_count += 1
        elif score < threshold and in_segment:
            if seg_count >= min_segment_length:
                detections.append({
                    "class_idx": class_idx,
                    "confidence": seg_conf_sum / max(seg_count, 1),
                    "start": seg_start,
                    "end": t - 1,
                })
            in_segment = False
            seg_conf_sum = 0.0
            seg_count = 0

    if in_segment and seg_count >= min_segment_length:
        detections.append({
            "class_idx": class_idx,
            "confidence": seg_conf_sum / max(seg_count, 1),
            "start": seg_start,
            "end": len(frame_scores) - 1,
        })
    return detections


def make_predictions_dict(outputs, num_classes, threshold=0.5):
    predictions = {}
    batch_size = outputs["M_fine"].shape[0]

    for b in range(batch_size):
        video_id = f"video_{b}"
        predictions[video_id] = {}
        M_fine_b = outputs["M_fine"][b]
        if hasattr(M_fine_b, 'detach'):
            M_fine_b = M_fine_b.detach().cpu().numpy()
        num_frames = M_fine_b.shape[0]
        for c in range(1, num_classes):
            frame_scores = M_fine_b[:, c]
            dets = extract_detections_from_scores(frame_scores, c, threshold=threshold)
            if dets:
                predictions[video_id][c] = dets
    return predictions


def make_ground_truths_dict(class_labels, num_frames_list, num_classes):
    ground_truths = {}
    if hasattr(class_labels, 'cpu'):
        cls_arr = class_labels.cpu().numpy()
    else:
        cls_arr = np.asarray(class_labels)
    for b, cls in enumerate(cls_arr):
        video_id = f"video_{b}"
        ground_truths[video_id] = {}
        nf = int(num_frames_list[b]) if hasattr(num_frames_list, '__getitem__') else 32
        if cls > 0:
            ground_truths[video_id][int(cls)] = [{
                "start": 0,
                "end": nf - 1,
            }]
    return ground_truths
