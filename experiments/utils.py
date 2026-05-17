from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: torch.nn.Module, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint_into(model: torch.nn.Module, path: Path, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu")
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


class StreamLogger:
    """Lightweight stdout logger that also appends to a JSONL file."""
    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(log_path, "a")
        else:
            self._fh = None
        self._t0 = time.time()

    def log(self, **fields: Any) -> None:
        line = " ".join(f"{k}={v}" for k, v in fields.items())
        elapsed = time.time() - self._t0
        msg = f"[{elapsed:7.1f}s] {line}"
        print(msg, file=sys.stdout, flush=True)
        if self._fh:
            json.dump({"t": elapsed, **fields}, self._fh, default=str)
            self._fh.write("\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "StreamLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
