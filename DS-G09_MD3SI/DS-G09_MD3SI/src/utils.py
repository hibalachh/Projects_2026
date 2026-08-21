from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg):
    for key in ["training_csv", "evaluation_csv", "model", "before_video", "after_video", "learning_curve", "evaluation_figure"]:
        Path(cfg["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["tensorboard_dir"]).mkdir(parents=True, exist_ok=True)
