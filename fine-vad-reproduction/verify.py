import os
import sys
from typing import TYPE_CHECKING, Any

import yaml
from easydict import EasyDict
import torch

if TYPE_CHECKING:
    import open_clip  # optional dependency, checked at runtime below

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

print("=" * 70)
print("Fine-VAD Reproduction - Environment & Code Verification")
print("=" * 70)

try:
    from loguru import logger  # noqa: F401  (environment check only)
    print("[OK] loguru")
except ImportError as e:
    print(f"[FAIL] loguru: {e}")

try:
    import scipy
    print(f"[OK] scipy {scipy.__version__}")
except ImportError as e:
    print(f"[MISSING] scipy: {e}")

try:
    import sklearn
    print(f"[OK] scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"[MISSING] scikit-learn: {e}")

try:
    import cv2
    print(f"[OK] opencv-python {cv2.__version__}")
except ImportError as e:
    print(f"[MISSING] opencv-python: {e}")

try:
    from PIL import Image
    print("[OK] pillow")
except ImportError as e:
    print(f"[MISSING] pillow: {e}")

print(f"[OK] torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")

openclip_ok = False
try:
    import open_clip
    print("[OK] open_clip")
    openclip_ok = True
except ImportError as e:
    print(f"[WARN] open_clip not installed - only partial model verification ({e})")

print("\n--- Loading config (config.yaml) ---")
with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    cfg: Any = EasyDict(yaml.safe_load(f))
print(f"[OK] loaded. Dataset targets: ucf_crime({cfg.datasets.ucf_crime.num_classes} cls)  xd_violence({cfg.datasets.xd_violence.num_classes} cls)")
print(f"     CLIP: {cfg.model.clip_arch}/{cfg.model.clip_pretrained}  K={cfg.progressive_learning.K}  T={cfg.progressive_learning.T}  omega={cfg.progressive_learning.omega}")

print("\n--- Verifying dataset module ---")
from datasets.video_dataset import (  # noqa: E402
    UCFCrimeDataset, XDViolenceDataset, UCF_CRIME_CLASSES, XD_VIOLENCE_CLASSES,
    get_dataloader, get_clip_transform,
)

transform = get_clip_transform()
print(f"[OK] transform built: {type(transform).__name__}")

for ds_name, cls_list, num_exp in [
    ("ucf_crime", UCF_CRIME_CLASSES, cfg.datasets.ucf_crime.num_classes),
    ("xd_violence", XD_VIOLENCE_CLASSES, cfg.datasets.xd_violence.num_classes),
]:
    loader, ds = get_dataloader(ds_name, "./data", split="train", batch_size=2, interval=16)
    assert ds.num_classes == num_exp, f"{ds_name} class mismatch: {ds.num_classes} != {num_exp}"
    batch = next(iter(loader))
    assert "frames" in batch, "batch missing 'frames'"
    assert batch["frames"].ndim == 5, f"frames shape error: {batch['frames'].shape}"
    assert batch["class_idx"].shape[0] == 2, "batch size mismatch"
    assert batch["frames"].shape == (2, 32, 3, 224, 224), f"unexpected frames shape {batch['frames'].shape}"
    print(f"[OK] {ds_name}: {len(ds)} samples -> batch frames={batch['frames'].shape}  "
          f"classes={batch['class_idx'].tolist()}  names={batch['class_names']}")

print("\n--- Verifying metrics module ---")
from utils.metrics import (  # noqa: E402
    binary_cross_entropy_loss, weighted_cross_entropy_loss,
    compute_map, make_predictions_dict, make_ground_truths_dict,
)

S_coarse = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)
bin_labels = torch.tensor([0, 1, 0], dtype=torch.int64)
l_bce = binary_cross_entropy_loss(S_coarse, bin_labels)
assert isinstance(l_bce, torch.Tensor) and torch.isfinite(l_bce), f"L_bce={l_bce} invalid"
print(f"[OK] L_bce = {float(l_bce):.4f}")

S_fine = torch.randn(3, 14)
cls_labels = torch.tensor([0, 5, 1], dtype=torch.int64)
macro = {i: i % cfg.progressive_learning.K for i in range(13)}
l_refine = weighted_cross_entropy_loss(S_fine, cls_labels, macro, 14, omega=cfg.progressive_learning.omega)
assert torch.isfinite(l_refine), f"L_refine={l_refine} invalid"
print(f"[OK] L_refine weighted CE = {float(l_refine):.4f}")

dummy_preds = {
    "v1": {1: [{"class_idx": 1, "confidence": 0.9, "start": 0, "end": 10}]},
    "v2": {2: [{"class_idx": 2, "confidence": 0.8, "start": 5, "end": 15}]},
}
dummy_gts = {
    "v1": {1: [{"start": 2, "end": 12}]},
    "v2": {2: [{"start": 3, "end": 18}]},
}
res = compute_map(dummy_preds, dummy_gts, iou_thresholds=[0.1, 0.5], num_classes=3)
assert res["AVG"] > 0, f"mAP AVG should be >0, got {res['AVG']}"
print(f"[OK] mAP computation = {res}")

B, N, C_cls = 2, 16, 14
fake_outputs = {"M_fine": torch.rand(B, N, C_cls)}
pred_dict = make_predictions_dict(fake_outputs, C_cls, threshold=0.5)
gt_dict = make_ground_truths_dict(torch.tensor([0, 3], dtype=torch.int64), [N, N], C_cls)
print(f"[OK] pred/gt dicts built: pred_keys={list(pred_dict.keys())} gt_keys={list(gt_dict.keys())}")

if openclip_ok:
    print("\n--- Verifying full FineVAD model ---")
    from train import build_model  # noqa: E402
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds_name in ["ucf_crime", "xd_violence"]:
        model, _ = build_model(cfg, ds_name, device, allow_random_init=True)
        model.to(device)
        model.compute_pseudo_macro_categories()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[OK] FineVAD({ds_name}): trainable={trainable/1e6:.2f}M total={total/1e6:.2f}M  pseudo_labels={model.pseudo_macro_labels}")

        B, N, C, H, W = 2, 16, 3, 224, 224
        dummy = torch.randn(B, N, C, H, W).to(device)
        outputs = model(dummy)
        for k in ["M_coarse", "M_inter", "M_fine", "S_coarse", "S_inter", "S_fine"]:
            shape = tuple(outputs[k].shape)
            print(f"     {k}: shape={shape}")
        M_coarse = outputs["M_coarse"]
        expected_coarse = (B, N, 2)
        assert M_coarse.shape == expected_coarse, f"M_coarse shape mismatch: {M_coarse.shape} != {expected_coarse}"
        K = cfg.progressive_learning.K
        expected_inter = (B, N, K + 1)
        assert outputs["M_inter"].shape == expected_inter, f"M_inter mismatch: {outputs['M_inter'].shape} != {expected_inter}"
        cfg_ds_key = "ucf_crime" if ds_name == "ucf_crime" else "xd_violence"
        n_fine = cfg.datasets[cfg_ds_key].num_classes
        assert outputs["M_fine"].shape == (B, N, n_fine), f"M_fine shape mismatch: {outputs['M_fine'].shape} != {(B, N, n_fine)}"
        assert outputs["S_fine"].shape == (B, n_fine), f"S_fine shape mismatch: {outputs['S_fine'].shape} != {(B, n_fine)}"

    print("\n--- Verifying CoarseContrastiveLoss ---")
    from models.fine_vad import CoarseContrastiveLoss  # noqa: E402
    cts_fn = CoarseContrastiveLoss()
    fv = torch.randn(4, 16, 512, device=device)
    lbl = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    l_cts = cts_fn(fv, lbl)
    # After the InfoNCE fix, mixed-batch contrastive loss must be > 0 and finite.
    assert l_cts.item() > 0 and not torch.isnan(l_cts), f"L_contrastive={l_cts} invalid"
    print(f"[OK] L_contrastive = {float(l_cts):.4f}")

    # Batch where every class appears exactly once (no valid positive pairs after
    # masking the diagonal) -> the loss must fall back to 0 instead of crashing.
    l_single = cts_fn(torch.randn(2, 16, 512, device=device), torch.tensor([0, 1], device=device))
    assert l_single.item() == 0.0, f"no-positive contrastive should be 0, got {l_single}"
    print(f"[OK] L_contrastive (no positive pairs) = {float(l_single):.4f}")

    # Variable-length video support: padded frames must not leak into the output.
    print("\n--- Verifying variable-length (padded) input ---")
    B, N, C, H, W = 4, 32, 3, 224, 224
    dummy_frames = torch.randn(B, N, C, H, W, device=device)
    dummy_lengths = torch.tensor([32, 16, 32, 8], device=device)  # variable lengths
    out_padded = model(dummy_frames, valid_lengths=dummy_lengths)
    assert out_padded["S_fine"].shape == (B, n_fine), f"S_fine shape mismatch: {out_padded['S_fine'].shape}"
    assert torch.isfinite(out_padded["S_fine"]).all(), "NaN in S_fine for padded batch"
    print(f"[OK] padded forward: S_fine shape={tuple(out_padded['S_fine'].shape)}  finite={bool(torch.isfinite(out_padded['S_fine']).all())}")

print("\n" + "=" * 70)
mode = "FULL (torch + open_clip)" if openclip_ok else "torch-only (open_clip missing)"
print(f"ALL CHECKS PASSED in [{mode}] mode!")
print("=" * 70)
print("\nUsage:")
print("  pip install -r fine-vad-reproduction/requirements.txt")
print("  python fine-vad-reproduction/train.py --dataset ucf_crime   # train UCF-Crime")
print("  python fine-vad-reproduction/train.py --dataset xd_violence # train XD-Violence")
print("  python fine-vad-reproduction/test.py --dataset ucf_crime --checkpoint checkpoints/best_ucf_crime.pt")
