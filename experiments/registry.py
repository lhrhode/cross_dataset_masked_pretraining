from __future__ import annotations

from dataclasses import fields, replace
from itertools import product
from typing import Any, Iterable, TypeVar

from .config import FinetuneConfig, PretrainConfig

_C = TypeVar("_C", PretrainConfig, FinetuneConfig)


CORE_DATASETS: tuple[str, ...] = ("ddti", "tn3k")
CORE_CONDITIONS: tuple[str, ...] = ("scratch", "within", "cross")
CORE_BUDGETS: tuple[int, ...] = (10, 50, 200)

# BUSI is the optional breast-ultrasound extension.
BUSI_DATASETS: tuple[str, ...] = ("busi",)
BUSI_BUDGETS: tuple[int, ...] = (10, 50, 200)

# Masking-size ablation (Table 6): repeated on TN3k @ N=50.
ABL_MASK_SIZES: tuple[int, ...] = (10, 20, 40, 80, 100)
ABL_MASK_RATIOS: tuple[float, ...] = (0.2, 0.6)
ABL_BUDGET: int = 50

# Masking-strategy ablation: keep one mask size/ratio combo.
STRAT_LIST: tuple[str, ...] = ("block", "random_pixel", "grid", "random_patch", "mixed")
STRAT_PATCH_SIZE: int = 40
STRAT_RATIO: float = 0.2
STRAT_BUDGET: int = 50

# Architecture comparison: which budgets/conditions to mirror with MS-UNet.
ARCH_BUDGETS: tuple[int, ...] = (10, 50, 200)
ARCH_CONDITIONS: tuple[str, ...] = ("scratch", "cross")
ARCH_DATASETS: tuple[str, ...] = ("ddti", "tn3k")

def _pretrain_split(dataset: str) -> str:
    # DDTI uses its 509-image train pool. TN3k and BUSI use the full pool we
    # have access to (BUSI's "all" excludes the held-out test partition).
    return "train" if dataset == "ddti" else "all"


def _finetune_split(dataset: str) -> str:
    return "train" if dataset == "ddti" else "all"

def _finetune_epochs(budget: int) -> int:
    # Scaled down ~5x from the paper (which uses 2000/1000/500 for 10/50/200)
    # so the full sweep stays tractable on one GPU.
    return {10: 400, 50: 250, 200: 150}[budget]


def _other(target: str) -> str:
    """For DDTI/TN3k pairings; BUSI cross-pretrains from TN3k (the larger pool)."""
    if target == "ddti":
        return "tn3k"
    if target == "tn3k":
        return "ddti"
    if target == "busi":
        return "tn3k"
    raise ValueError(target)


# ---------------------------------------------------------------------------
# Core: 30 transfer experiments (5 budgets × 3 conditions × 2 datasets)
# ---------------------------------------------------------------------------
def _build_core_pretrains() -> dict[str, PretrainConfig]:
    out: dict[str, PretrainConfig] = {}
    for ds in CORE_DATASETS:
        name = f"pretrain_{ds}"
        out[name] = PretrainConfig(
            name=name, dataset=ds, 
            split=_pretrain_split(ds), 
            epochs=200, lr=1e-3,
        )
    return out


def _build_core_finetunes() -> dict[str, FinetuneConfig]:
    out: dict[str, FinetuneConfig] = {}
    for target in CORE_DATASETS:
        other = _other(target)
        for cond in CORE_CONDITIONS:
            pre = (
                None if cond == "scratch"
                else f"pretrain_{target if cond == 'within' else other}"
            )
            for budget in CORE_BUDGETS:
                name = f"{target}_{cond}_n{budget}"
                out[name] = FinetuneConfig(
                    name=name, target_dataset=target,
                    n_labeled=budget, condition=cond,
                    pretrain_name=pre,
                    train_split=_finetune_split(target),
                    epochs=_finetune_epochs(budget), lr=1e-3,
                )
    return out

def _build_busi_pretrains() -> dict[str, PretrainConfig]:
    name = "pretrain_busi"
    return {
        name: PretrainConfig(
            name=name, dataset="busi",
            split="all", epochs=200, lr=1e-3,
        )
    }


def _build_busi_finetunes() -> dict[str, FinetuneConfig]:
    out: dict[str, FinetuneConfig] = {}
    target = "busi"
    other = _other(target)  # tn3k
    for cond in CORE_CONDITIONS:
        pre = (
            None if cond == "scratch"
            else f"pretrain_{target if cond == 'within' else other}"
        )
        for budget in BUSI_BUDGETS:
            name = f"{target}_{cond}_n{budget}"
            out[name] = FinetuneConfig(
                name=name, target_dataset=target, 
                n_labeled=budget, condition=cond,
                pretrain_name=pre,
                train_split=_finetune_split(target),
                epochs=_finetune_epochs(budget), lr=1e-3,
            )
    return out


def _build_mask_ablation() -> tuple[dict[str, PretrainConfig], dict[str, FinetuneConfig]]:
    pre: dict[str, PretrainConfig] = {}
    fin: dict[str, FinetuneConfig] = {}
    for p, r in product(ABL_MASK_SIZES, ABL_MASK_RATIOS):
        tag = f"p{p}_r{int(r * 10):02d}"
        pname = f"pretrain_tn3k_{tag}"
        pre[pname] = PretrainConfig(
            name=pname, dataset="tn3k", split="all",
            epochs=100, lr=1e-3,
            patch_size=p, mask_ratio=r, mask_strategy="block",
        )
        # Downstream: DDTI N=50 cross-pretrained on TN3k.
        fname = f"ddti_cross_n{ABL_BUDGET}_{tag}"
        fin[fname] = FinetuneConfig(
            name=fname, target_dataset="ddti",
            n_labeled=ABL_BUDGET, condition="cross",
            pretrain_name=pname, train_split="train",
            epochs=_finetune_epochs(ABL_BUDGET), lr=1e-3,
        )
    return pre, fin

def _build_strategy_ablation() -> tuple[dict[str, PretrainConfig], dict[str, FinetuneConfig]]:
    pre: dict[str, PretrainConfig] = {}
    fin: dict[str, FinetuneConfig] = {}
    for strat in STRAT_LIST:
        tag = f"s{strat}"
        pname = f"pretrain_tn3k_{tag}"
        pre[pname] = PretrainConfig(
            name=pname, dataset="tn3k", split="all",
            epochs=100, lr=1e-3,
            patch_size=STRAT_PATCH_SIZE, mask_ratio=STRAT_RATIO,
            mask_strategy=strat,
        )
        fname = f"ddti_cross_n{STRAT_BUDGET}_{tag}"
        fin[fname] = FinetuneConfig(
            name=fname, target_dataset="ddti", 
            n_labeled=STRAT_BUDGET, condition="cross", 
            pretrain_name=pname, train_split="train",
            epochs=_finetune_epochs(STRAT_BUDGET), lr=1e-3,
        )
    return pre, fin


# ---------------------------------------------------------------------------
# Architecture sweep (MS-UNet mirror of a subset of the core matrix)
# ---------------------------------------------------------------------------
def _build_arch_sweep() -> tuple[dict[str, PretrainConfig], dict[str, FinetuneConfig]]:
    pre: dict[str, PretrainConfig] = {}
    fin: dict[str, FinetuneConfig] = {}
    # MS-UNet pretrain on each dataset.
    for ds in ARCH_DATASETS:
        pname = f"pretrain_{ds}_msunet"
        pre[pname] = PretrainConfig(
            name=pname, dataset=ds, 
            split=_pretrain_split(ds), 
            epochs=200, lr=1e-3, architecture="msunet",
        )
    # Downstream finetunes: scratch vs cross at each budget.
    for target in ARCH_DATASETS:
        other = _other(target)
        for cond in ARCH_CONDITIONS:
            pre_name = None if cond == "scratch" else f"pretrain_{other}_msunet"
            for budget in ARCH_BUDGETS:
                fname = f"{target}_{cond}_n{budget}_msunet"
                fin[fname] = FinetuneConfig(
                    name=fname, target_dataset=target, 
                    n_labeled=budget, condition=cond, 
                    pretrain_name=pre_name,
                    train_split=_finetune_split(target), 
                    epochs=_finetune_epochs(budget), lr=1e-3,
                    architecture="msunet",
                )
    return pre, fin


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------
CORE_PRETRAINS = _build_core_pretrains()
CORE_FINETUNES = _build_core_finetunes()
BUSI_PRETRAINS = _build_busi_pretrains()
BUSI_FINETUNES = _build_busi_finetunes()
ABL_PRETRAINS, ABL_FINETUNES = _build_mask_ablation()
STRAT_PRETRAINS, STRAT_FINETUNES = _build_strategy_ablation()
ARCH_PRETRAINS, ARCH_FINETUNES = _build_arch_sweep()


def all_pretrain_configs() -> dict[str, PretrainConfig]:
    out: dict[str, PretrainConfig] = {}
    for src in (CORE_PRETRAINS, BUSI_PRETRAINS, ABL_PRETRAINS, STRAT_PRETRAINS, ARCH_PRETRAINS):
        out.update(src)
    return out


def all_finetune_configs() -> dict[str, FinetuneConfig]:
    out: dict[str, FinetuneConfig] = {}
    for src in (CORE_FINETUNES, BUSI_FINETUNES, ABL_FINETUNES, STRAT_FINETUNES, ARCH_FINETUNES):
        out.update(src)
    return out


SECTIONS: dict[str, dict] = {
    "core": {"pretrain": CORE_PRETRAINS, "finetune": CORE_FINETUNES},
    "busi": {"pretrain": BUSI_PRETRAINS, "finetune": BUSI_FINETUNES},
    "mask_size_ablation": {"pretrain": ABL_PRETRAINS, "finetune": ABL_FINETUNES},
    "mask_strategy_ablation": {"pretrain": STRAT_PRETRAINS, "finetune": STRAT_FINETUNES},
    "architecture": {"pretrain": ARCH_PRETRAINS, "finetune": ARCH_FINETUNES},
}


def section_names() -> Iterable[str]:
    return SECTIONS.keys()


def collect(sections: Iterable[str]) -> tuple[dict[str, PretrainConfig], dict[str, FinetuneConfig]]:
    pre: dict[str, PretrainConfig] = {}
    fin: dict[str, FinetuneConfig] = {}
    for s in sections:
        if s not in SECTIONS:
            raise KeyError(f"Unknown section {s!r}. Available: {list(SECTIONS)}")
        pre.update(SECTIONS[s]["pretrain"])
        fin.update(SECTIONS[s]["finetune"])
    return pre, fin

PRETRAIN_CONFIGS: dict[str, PretrainConfig] = all_pretrain_configs()
FINETUNE_CONFIGS: dict[str, FinetuneConfig] = all_finetune_configs()


def apply_overrides(cfg: _C, overrides: dict[str, Any] | None) -> _C:
    """Return a copy of ``cfg`` with ``overrides`` applied.

    Raises ``KeyError`` if an override names a field that does not exist on
    the config dataclass — catches typos in ``--set foo=bar`` before training.
    """
    if not overrides:
        return cfg
    valid = {f.name for f in fields(cfg)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise KeyError(
            f"Unknown override field(s) for {type(cfg).__name__}: {unknown}. "
            f"Valid fields: {sorted(valid)}"
        )
    return replace(cfg, **overrides)

