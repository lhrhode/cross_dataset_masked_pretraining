from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

DatasetName = Literal["ddti", "tn3k", "busi"]
SplitName = Literal["train", "val", "test", "all"]
ArchName = Literal["unet", "msunet"]
MaskStrategyName = Literal["block", "random_pixel", "grid", "random_patch", "mixed"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class PretrainConfig:
    #Masked-reconstruction pretraining on the FULL train pool of a dataset
    name: str
    dataset: DatasetName
    split: SplitName = "train"  # DDTI: "train" (509). TN3k: override to "all".
    image_size: int = 256
    batch_size: int = 10
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    mask_ratio: float = 0.20
    patch_size: int = 40
    mask_strategy: MaskStrategyName = "block"
    patch_size_b: int = 10
    base_channels: int = 32
    architecture: ArchName = "unet"
    num_workers: int = 4
    seed: int = 0
    runs_dir: Path = field(default_factory=lambda: DEFAULT_RUNS_DIR)
    log_every: int = 50  # iterations
    save_last_only: bool = True

    @property
    def out_dir(self) -> Path:
        return Path(self.runs_dir) / self.name

    @property
    def checkpoint_path(self) -> Path:
        return self.out_dir / "checkpoint.pt"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runs_dir"] = str(self.runs_dir)
        return d


@dataclass
class FinetuneConfig:
    #Supervised segmentation fine-tuning with a few-label budget

    name: str
    target_dataset: DatasetName
    n_labeled: int
    condition: Literal["scratch", "within", "cross"]
    pretrain_name: str | None = None

    train_split: SplitName = "train"  # subsampled to n_labeled
    test_split: SplitName = "test"
    image_size: int = 256
    batch_size: int = 10
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    base_channels: int = 32
    architecture: ArchName = "unet"
    num_workers: int = 4
    seed: int = 0
    fold: int = 0  # TN3k only
    dice_weight: float = 1.0  # BCE + dice_weight * SoftDice
    runs_dir: Path = field(default_factory=lambda: DEFAULT_RUNS_DIR)
    log_every: int = 50  # iterations
    eval_every: int = 0  # epochs; 0 disables periodic eval (final-only)

    @property
    def out_dir(self) -> Path:
        return Path(self.runs_dir) / self.name

    @property
    def metrics_path(self) -> Path:
        return self.out_dir / "metrics.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.out_dir / "checkpoint.pt"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runs_dir"] = str(self.runs_dir)
        return d
