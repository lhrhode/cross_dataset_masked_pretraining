from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader

@torch.no_grad()
def dice_score(pred_probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Per-sample Dice between sigmoid probs (thresholded at 0.5) and {0,1} target
    pred = (pred_probs > 0.5).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (2 * inter + eps) / (denom + eps)


@torch.no_grad()
def iou_score(pred_probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Per-sample IoU (Jaccard) between thresholded preds and {0,1} target
    pred = (pred_probs > 0.5).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
    return (inter + eps) / (union + eps)


@torch.no_grad()
def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Differentiable soft Dice loss for training
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def _surface(mask: np.ndarray) -> np.ndarray:
    # Boolean array marking the 1-pixel boundary of `mask` (1 = surface)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def hd95_single(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")

    pred_surf = _surface(pred)
    target_surf = _surface(target)
    # Distance from every pixel to the nearest surface pixel of the *other* mask.
    dt_to_target = distance_transform_edt(~target_surf)
    dt_to_pred = distance_transform_edt(~pred_surf)
    distances = np.concatenate([dt_to_target[pred_surf], dt_to_pred[target_surf]])
    return float(np.percentile(distances, 95))


@torch.no_grad()
def hd95_batch(pred_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample HD95 for a batch of (B, 1, H, W) tensors."""
    pred_np = (pred_probs > 0.5).cpu().numpy().astype(bool)[:, 0]
    tgt_np = target.cpu().numpy().astype(bool)[:, 0]
    out = np.array([hd95_single(p, t) for p, t in zip(pred_np, tgt_np)], dtype=np.float32)
    return torch.from_numpy(out)

def boundary_f1_single(pred: np.ndarray, target: np.ndarray, tol: int = 2) -> float:
    # Symmetric boundary F1
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    if not pred.any() or not target.any():
        return 0.0
    pred_surf = _surface(pred)
    target_surf = _surface(target)
    iters = max(1, int(tol))
    pred_band = binary_dilation(pred_surf, iterations=iters)
    target_band = binary_dilation(target_surf, iterations=iters)
    n_pred = int(pred_surf.sum())
    n_tgt = int(target_surf.sum())
    if n_pred == 0 or n_tgt == 0:
        return 0.0
    tp_pred = int((pred_surf & target_band).sum())
    tp_tgt = int((target_surf & pred_band).sum())
    precision = tp_pred / max(1, n_pred)
    recall = tp_tgt / max(1, n_tgt)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


@torch.no_grad()
def bf1_batch(pred_probs: torch.Tensor, target: torch.Tensor, tol: int = 2) -> torch.Tensor:
    pred_np = (pred_probs > 0.5).cpu().numpy().astype(bool)[:, 0]
    tgt_np = target.cpu().numpy().astype(bool)[:, 0]
    out = np.array([boundary_f1_single(p, t, tol=tol) for p, t in zip(pred_np, tgt_np)],
                   dtype=np.float32)
    return torch.from_numpy(out)


# Full evaluation loop
def _summary(values: torch.Tensor, name: str) -> dict:
    arr = values.numpy().astype(np.float64) if values.numel() else np.zeros(0)
    if arr.size == 0:
        return {f"{name}_mean": float("nan"), f"{name}_std": 0.0, f"{name}_n_valid": 0}
    finite = arr[np.isfinite(arr)]
    mean = float(np.mean(finite)) if finite.size else float("nan")
    std = float(np.std(finite, ddof=0)) if finite.size > 1 else 0.0
    return {f"{name}_mean": mean, f"{name}_std": std, f"{name}_n_valid": int(finite.size)}


@torch.no_grad()
def evaluate_segmentation(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    # Compute Dice, IoU, and HD95 over the entire loader.
    model.eval()
    dices: list[torch.Tensor] = []
    ious: list[torch.Tensor] = []
    hd95s: list[torch.Tensor] = []
    bf1s: list[torch.Tensor] = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        logits = model(img)
        probs = torch.sigmoid(logits)
        dices.append(dice_score(probs, seg).cpu())
        ious.append(iou_score(probs, seg).cpu())
        hd95s.append(hd95_batch(probs, seg))
        bf1s.append(bf1_batch(probs, seg))
    dices_t = torch.cat(dices) if dices else torch.zeros(0)
    ious_t = torch.cat(ious) if ious else torch.zeros(0)
    hd95s_t = torch.cat(hd95s) if hd95s else torch.zeros(0)
    bf1s_t = torch.cat(bf1s) if bf1s else torch.zeros(0)

    out = {"n": int(dices_t.numel())}
    out.update(_summary(dices_t, "dice"))
    out.update(_summary(ious_t, "iou"))
    out.update(_summary(hd95s_t, "hd95"))
    out.update(_summary(bf1s_t, "bf1"))
    out["per_sample"] = {
        "dice": dices_t.tolist(),
        "iou": ious_t.tolist(),
        "hd95": hd95s_t.tolist(),
        "bf1": bf1s_t.tolist(),
    }
    return out

def evaluate_dice(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    res = evaluate_segmentation(model, loader, device)
    return {"dice_mean": res["dice_mean"], "dice_std": res["dice_std"], "n": res["n"]}
