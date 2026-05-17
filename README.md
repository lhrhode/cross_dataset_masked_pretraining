# Cross-Dataset Masked-Pretraining for Ultrasound Segmentation

Few-label segmentation transfer study on thyroid and breast ultrasound. Pretrains
a U-Net (or lite MS-UNet) on masked reconstruction of one dataset, then fine-tunes
on a small labeled budget of a second dataset.

The experimental matrix follows the protocol in *He et al. 2025* for three transfer
conditions (`scratch` / `within` / `cross`) at three label budgets (N ∈ {10, 50, 200})
across DDTI, TN3k, and (optionally) BUSI, plus masking and architecture ablations.

## Repository layout

```
code/
├── data/
│   └── thyroid_dataset.py     # DDTI / TN3k / BUSI loaders + masking transforms
└── experiments/
    ├── config.py              # PretrainConfig, FinetuneConfig dataclasses
    ├── registry.py            # Every named experiment, grouped into 5 sections
    ├── run.py                 # `python -m experiments.run --name <name>`
    ├── pretrain.py            # Masked-reconstruction training loop
    ├── finetune.py            # Supervised segmentation training loop
    ├── evaluate.py            # Dice / IoU / HD95 / Boundary-F1
    ├── model.py               # U-Net + architecture dispatcher
    ├── model_msunet.py        # Lite MS-UNet variant
    └── utils.py               # Seeding, checkpoint I/O, JSON logger
```

## Setup

### Dependencies

Python ≥ 3.10. Install (in a venv / conda env):

```bash
pip install torch numpy scipy pillow
```

### Dataset paths

The dataset loader resolves three roots, two of which are configurable via env vars:

| Dataset | Default path                                                       | Override         |
|---------|--------------------------------------------------------------------|------------------|
| DDTI    | `$THYROID_ROOT/DDTI/2_preprocessed_data/stage1/{p_image,p_mask}`   | `THYROID_ROOT`   |
| TN3k    | `$THYROID_ROOT/tn3k/{trainval,test}-{image,mask}` + fold JSONs     | `THYROID_ROOT`   |
| BUSI    | `$BUSI_ROOT/{benign,malignant}/`                                   | `BUSI_ROOT`      |

The default `THYROID_ROOT` is the lab path
`Thyroid_US/Thyroid Dataset/`. Set the env var
if you're running anywhere else.

Unzip Thyroid_US.zip and edit dataset paths accordingly.

## Usage

```bash
python -m experiments.run --name <experiment_name>
```

Pretrain checkpoints are required by `within`- and `cross`-condition finetunes;
the runner does not pretrain first, then finetune.

### Examples

```bash
# 1. Pretrain on TN3k (masked reconstruction)
python -m experiments.run --name pretrain_tn3k

# 2. Fine-tune DDTI with 50 labels, initialized from the TN3k pretrain
python -m experiments.run --name ddti_cross_n50

# 3. From-scratch baseline at the same budget
python -m experiments.run --name ddti_scratch_n50

# 4. Override
python -m experiments.run --name pretrain_tn3k --set epochs=5 lr=5e-4

# 5. Pin device
python -m experiments.run --name ddti_cross_n50 --device cuda:1
```

## Experiment registry

`experiments/registry.py` builds five sections lookup tables `PRETRAIN_CONFIGS` / `FINETUNE_CONFIGS`:

| Section                  | Pretrains | Finetunes | Purpose                                            |
|--------------------------|-----------|-----------|----------------------------------------------------|
| `core`                   | 2         | 18        | 3 budgets × 3 conditions × 2 datasets (DDTI, TN3k) |
| `busi`                   | 1         | 9         | Breast-ultrasound extension                        |
| `mask_size_ablation`     | 10        | 10        | 5 patch sizes × 2 mask ratios on TN3k → DDTI       |
| `mask_strategy_ablation` | 5         | 5         | block / random_pixel / grid / random_patch / mixed |
| `architecture`           | 2         | 12        | MS-UNet mirror of a subset of core                 |

## Outputs

Each run writes to `runs/<experiment_name>/`:

- `checkpoint.pt` — model weights (`save_last_only=True` by default).
- `metrics.json` — final Dice/IoU/HD95/Boundary-F1 mean/std + per-sample (finetunes).
- `log.jsonl` — per-iteration training log (one JSON object per line).
- `config.json` — the resolved `PretrainConfig` / `FinetuneConfig`.

`run.py` also prints a one-line summary to stdout on completion.

