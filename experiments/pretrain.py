from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sibling packages importable when run as a script (python pretrain.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.thyroid_dataset import MaskingConfig, build_dataloader

from .config import PretrainConfig
from .model import build_model
from .utils import StreamLogger, pick_device, save_checkpoint, set_seed, write_json


def _masked_mse(logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    sq = (pred - target) ** 2 * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return sq.sum() / denom


def run_pretrain(cfg: PretrainConfig, device: torch.device | None = None) -> dict:
    # Execute one pretraining run. Returns summary dict, writes checkpoint
    device = device or pick_device()
    set_seed(cfg.seed)

    masking = MaskingConfig(
        patch_size=cfg.patch_size,
        ratio=cfg.mask_ratio,
        strategy=cfg.mask_strategy,
        patch_size_b=cfg.patch_size_b,
    )
    loader = build_dataloader(
        name=cfg.dataset,
        split=cfg.split,
        mode="pretrain",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        masking=masking,
        seed=cfg.seed,
    )

    model = build_model(
        architecture=cfg.architecture, base_ch=cfg.base_channels,
        in_ch=1, out_ch=1,
    ).to(device)
    opt = torch.optim.Adamax(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    log = StreamLogger(cfg.out_dir / "train.log.jsonl")
    log.log(event="pretrain_start", name=cfg.name, dataset=cfg.dataset, n_iters_per_epoch=len(loader))

    model.train()
    iter_count = 0
    t0 = time.time()
    last_loss = float("nan")
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            m = batch["loss_mask"].to(device, non_blocking=True)
            logits = model(x)
            loss = _masked_mse(logits, tgt, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            iter_count += 1
            if iter_count % cfg.log_every == 0:
                log.log(event="iter", epoch=epoch, iter=iter_count, loss=float(loss.detach()))
        last_loss = epoch_loss / max(1, len(loader))
        log.log(event="epoch", epoch=epoch, avg_loss=last_loss)

    elapsed = time.time() - t0
    save_checkpoint(
        cfg.checkpoint_path,
        model,
        extra={"config": cfg.to_dict(), "final_avg_loss": last_loss, "elapsed_s": elapsed},
    )
    summary = {
        "name": cfg.name,
        "dataset": cfg.dataset,
        "epochs": cfg.epochs,
        "final_avg_loss": last_loss,
        "elapsed_s": elapsed,
        "checkpoint": str(cfg.checkpoint_path),
    }
    write_json(cfg.out_dir / "summary.json", summary)
    log.log(event="pretrain_done", **summary)
    log.close()
    return summary
