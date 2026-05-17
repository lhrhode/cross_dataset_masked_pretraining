from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.thyroid_dataset import build_dataloader

from .config import FinetuneConfig, PretrainConfig
from .evaluate import evaluate_segmentation, soft_dice_loss
from .model import build_model
from .utils import (
    StreamLogger,
    load_checkpoint_into,
    pick_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def _init_from_pretrain(model: nn.Module, pretrain_ckpt: Path) -> dict:
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(
            f"Pretrain checkpoint not found: {pretrain_ckpt}. Run the pretrain "
            f"experiment first."
        )
    return load_checkpoint_into(model, pretrain_ckpt, strict=True)


def run_finetune(
    cfg: FinetuneConfig,
    pretrain_cfg: PretrainConfig | None = None,
    device: torch.device | None = None,
) -> dict:
    """Execute one finetune run. Returns metrics dict; writes checkpoint."""
    device = device or pick_device()
    set_seed(cfg.seed)

    train_loader = build_dataloader(
        name=cfg.target_dataset,
        split=cfg.train_split,
        mode="finetune",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        n_labeled=cfg.n_labeled,
        fold=cfg.fold,
        seed=cfg.seed,
    )
    test_loader = build_dataloader(
        name=cfg.target_dataset,
        split=cfg.test_split,
        mode="finetune",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        shuffle=False,
        drop_last=False,
        augment=False,
        fold=cfg.fold,
        seed=cfg.seed,
    )

    model = build_model(
        architecture=cfg.architecture, base_ch=cfg.base_channels,
        in_ch=1, out_ch=1,
    ).to(device)
    init_info = {}
    if cfg.condition != "scratch":
        if pretrain_cfg is None:
            raise ValueError(
                f"finetune condition={cfg.condition} requires a pretrain_cfg "
                f"(got None) for experiment {cfg.name!r}"
            )
        init_info = _init_from_pretrain(model, pretrain_cfg.checkpoint_path)

    opt = torch.optim.Adamax(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    log = StreamLogger(cfg.out_dir / "train.log.jsonl")
    log.log(
        event="finetune_start",
        name=cfg.name,
        target=cfg.target_dataset,
        n_labeled=cfg.n_labeled,
        condition=cfg.condition,
        pretrain=cfg.pretrain_name,
        n_train=len(train_loader.dataset),
        n_test=len(test_loader.dataset),
        init_missing=len(init_info.get("missing", [])),
        init_unexpected=len(init_info.get("unexpected", [])),
    )

    def _slim(m: dict) -> dict:
        """Drop the per-sample arrays from a metrics dict for in-loop logging."""
        return {k: v for k, v in m.items() if k != "per_sample"}

    t0 = time.time()
    iter_count = 0
    best_dice = -1.0
    epoch_metrics: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            logits = model(img)
            loss = bce(logits, seg) + cfg.dice_weight * soft_dice_loss(logits, seg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            iter_count += 1
            if iter_count % cfg.log_every == 0:
                log.log(event="iter", epoch=epoch, iter=iter_count, loss=float(loss.detach()))
        avg = epoch_loss / max(1, len(train_loader))
        log.log(event="epoch", epoch=epoch, avg_loss=avg)
        if cfg.eval_every and (epoch + 1) % cfg.eval_every == 0:
            metrics = evaluate_segmentation(model, test_loader, device)
            slim = _slim(metrics)
            slim["epoch"] = epoch
            epoch_metrics.append(slim)
            best_dice = max(best_dice, metrics["dice_mean"])
            log.log(event="eval", **slim)

    final_metrics = evaluate_segmentation(model, test_loader, device)
    final_slim = _slim(final_metrics)
    final_slim["epoch"] = cfg.epochs - 1
    epoch_metrics.append(final_slim)
    best_dice = max(best_dice, final_metrics["dice_mean"])

    elapsed = time.time() - t0
    save_checkpoint(
        cfg.checkpoint_path,
        model,
        extra={"config": cfg.to_dict(), "final_dice": final_metrics["dice_mean"], "elapsed_s": elapsed},
    )
    out = {
        "name": cfg.name,
        "target_dataset": cfg.target_dataset,
        "n_labeled": cfg.n_labeled,
        "condition": cfg.condition,
        "pretrain_name": cfg.pretrain_name,
        "final_dice_mean": final_metrics["dice_mean"],
        "final_dice_std": final_metrics["dice_std"],
        "final_iou_mean": final_metrics["iou_mean"],
        "final_iou_std": final_metrics["iou_std"],
        "final_hd95_mean": final_metrics["hd95_mean"],
        "final_hd95_std": final_metrics["hd95_std"],
        "final_hd95_n_valid": final_metrics["hd95_n_valid"],
        "final_bf1_mean": final_metrics.get("bf1_mean"),
        "final_bf1_std": final_metrics.get("bf1_std"),
        "best_dice_mean": best_dice,
        "n_test": final_metrics["n"],
        "elapsed_s": elapsed,
        "epoch_metrics": epoch_metrics,
        "per_sample": final_metrics["per_sample"],
        "config": cfg.to_dict(),
    }
    write_json(cfg.metrics_path, out)
    log.log(
        event="finetune_done",
        dice=final_metrics["dice_mean"],
        iou=final_metrics["iou_mean"],
        hd95=final_metrics["hd95_mean"],
        elapsed_s=elapsed,
    )
    log.close()
    return out
