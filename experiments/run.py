# Usage `python -m experiments.run --name <experiment_name>`
# Runs pretrain or finetune based on the registry and config.

from __future__ import annotations

import argparse
import sys

from .finetune import run_finetune
from .pretrain import run_pretrain
from .registry import FINETUNE_CONFIGS, PRETRAIN_CONFIGS, apply_overrides


def _parse_overrides(items: list[str] | None) -> dict:
    out: dict = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one experiment by name.")
    parser.add_argument("--name", required=True, help="Experiment name from the registry.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Field overrides, e.g. --set epochs=5 lr=5e-4",
    )
    parser.add_argument("--device", default=None, help="Override torch device (e.g. cuda:0).")
    args = parser.parse_args(argv)

    overrides = _parse_overrides(args.set)

    if args.name in PRETRAIN_CONFIGS:
        cfg = apply_overrides(PRETRAIN_CONFIGS[args.name], overrides)
        summary = run_pretrain(cfg, device=None if args.device is None else _to_device(args.device))
        print(summary)
        return 0
    if args.name in FINETUNE_CONFIGS:
        cfg = apply_overrides(FINETUNE_CONFIGS[args.name], overrides)
        pretrain_cfg = (
            PRETRAIN_CONFIGS[cfg.pretrain_name] if cfg.pretrain_name else None
        )
        metrics = run_finetune(
            cfg, pretrain_cfg=pretrain_cfg, device=None if args.device is None else _to_device(args.device)
        )
        print({k: metrics[k] for k in ("name", "final_dice_mean", "final_dice_std", "elapsed_s")})
        return 0

    avail = ", ".join(list(PRETRAIN_CONFIGS) + list(FINETUNE_CONFIGS))
    print(f"Unknown experiment {args.name!r}. Available: {avail}", file=sys.stderr)
    return 2


def _to_device(s: str):
    import torch
    return torch.device(s)


if __name__ == "__main__":
    main()
