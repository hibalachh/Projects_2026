"""Dataset reading, validation and cleaning (no Streamlit imports).

The cleaning mirrors Project.ipynb: infinite/missing values removed, exact duplicate
rows removed, and the redundant ``Subflow Fwd Bytes`` column dropped.

``notebook_split`` recreates the notebook's train / threshold / test partition so the
app can evaluate the models on flows they never saw during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LABEL_COL, REDUNDANT_COL, REFERENCE_SPLIT, SEED

RAW_LABEL_COL = "Label"  # column name in the original CICIDS2017 CSVs ("BENIGN", "DoS", ...)


class DatasetError(ValueError):
    """Raised with a message that is safe to show to the person using the app."""


@dataclass
class Dataset:
    name: str
    df: pd.DataFrame                 # features (+ label), cleaned and de-duplicated
    features: list[str]
    has_labels: bool
    report: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable identifier used as a cache key."""
        return f"{self.name}|{len(self.df)}|{self.report.get('rows_raw', 0)}"

    @property
    def has_both_classes(self) -> bool:
        return self.has_labels and self.df[LABEL_COL].nunique() == 2

    def sample(self, n: int, seed: int = SEED) -> pd.DataFrame:
        """Random sample in original file order (the index keeps the source row number)."""
        n = min(int(n), len(self.df))
        if n >= len(self.df):
            return self.df
        return self.df.sample(n=n, random_state=seed).sort_index()

    def memory_mb(self) -> float:
        return float(self.df.memory_usage(deep=True).sum()) / 1_048_576


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_csv(source: str | Path | BinaryIO, features: list[str]) -> pd.DataFrame:
    """Read only the columns the app needs; strip whitespace from column names.

    Raw CICIDS2017 files have names like ``" Destination Port"`` with a leading space.
    """
    wanted = set(features) | {LABEL_COL, RAW_LABEL_COL, REDUNDANT_COL}
    try:
        raw = pd.read_csv(source, usecols=lambda c: str(c).strip() in wanted, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(
            "The file could not be parsed as a comma-separated CSV. "
            "Check that it is a plain-text CSV with a header row."
        ) from exc
    except ValueError as exc:  # e.g. usecols matched nothing
        raise DatasetError(f"The file could not be read: {exc}") from exc
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


# --------------------------------------------------------------------------- #
# Validation + cleaning
# --------------------------------------------------------------------------- #
def prepare_dataset(raw: pd.DataFrame, features: list[str], name: str) -> Dataset:
    """Validate the schema and clean the frame the way the notebook did."""
    missing = [c for c in features if c not in raw.columns]
    if missing:
        shown = ", ".join(f"“{c}”" for c in missing[:6])
        more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
        raise DatasetError(
            f"The CSV is missing {len(missing)} of the {len(features)} columns the models were "
            f"trained on: {shown}{more}. Upload a file with CICIDS2017-style flow features."
        )

    report: dict = {"rows_raw": int(len(raw))}
    keep = list(features) + ([REDUNDANT_COL] if REDUNDANT_COL in raw.columns else [])
    df = raw[keep]
    numeric = [pd.api.types.is_numeric_dtype(t) for t in df.dtypes]
    if not all(numeric):
        df = df.apply(pd.to_numeric, errors="coerce")
    else:
        df = df.copy()

    # ---- labels (optional) ---------------------------------------------------
    has_labels = False
    label_note = None
    if LABEL_COL in raw.columns:
        y = pd.to_numeric(raw[LABEL_COL], errors="coerce")
    elif RAW_LABEL_COL in raw.columns:
        text = raw[RAW_LABEL_COL].astype(str).str.strip().str.upper()
        y = (text != "BENIGN").astype("int8")
        label_note = "Labels derived from the “Label” column (BENIGN → 0, everything else → 1)."
    else:
        y = None
    if y is not None:
        valid = set(pd.unique(y.dropna())) <= {0, 1, 0.0, 1.0}
        if valid:
            df[LABEL_COL] = y.to_numpy()
            has_labels = True
        else:
            label_note = (f"“{LABEL_COL}” was ignored because it contains values other than 0 and 1.")
    report["label_note"] = label_note

    # ---- cleaning, in the same order as the notebook -----------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    report["dropped_nan_inf"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    if REDUNDANT_COL in df.columns:
        df = df.drop(columns=[REDUNDANT_COL])
    if has_labels:
        df[LABEL_COL] = df[LABEL_COL].astype("int8")

    if df.empty:
        raise DatasetError("No usable rows remain after removing missing and infinite values.")

    report["rows_clean"] = int(len(df))
    return Dataset(name=name, df=df, features=list(features), has_labels=has_labels, report=report)


def load_dataset(source: str | Path | BinaryIO, features: list[str], name: str) -> Dataset:
    return prepare_dataset(read_csv(source, features), features, name)


# --------------------------------------------------------------------------- #
# Notebook split
# --------------------------------------------------------------------------- #
@dataclass
class HoldoutSplit:
    """Positional row indices into ``Dataset.df`` for each partition."""
    train_idx: np.ndarray
    thresh_idx: np.ndarray
    test_idx: np.ndarray
    counts: dict[str, int]
    matches_notebook: bool


def notebook_split(ds: Dataset) -> HoldoutSplit:
    """Recreate the partition made in Project.ipynb (cell 82).

    * normal flows -> 70 % train / 15 % threshold-finding / 15 % test
    * attack flows -> 30 % threshold-finding / 70 % test

    ``train_test_split`` only depends on the number of rows and ``random_state``, so
    splitting positional index arrays gives exactly the rows the notebook obtained,
    without copying the full frame.
    """
    if not ds.has_both_classes:
        raise DatasetError("A held-out split needs labelled data containing both normal and attack flows.")

    y = ds.df[LABEL_COL].to_numpy()
    normal_pos = np.flatnonzero(y == 0)
    attack_pos = np.flatnonzero(y == 1)

    train_l, temp_l = train_test_split(np.arange(len(normal_pos)), test_size=0.30, random_state=SEED)
    thresh_n_l, test_n_l = train_test_split(temp_l, test_size=0.50, random_state=SEED)
    thresh_a_l, test_a_l = train_test_split(np.arange(len(attack_pos)), test_size=0.70, random_state=SEED)

    counts = {
        "train_normal": len(train_l),
        "thresh_normal": len(thresh_n_l),
        "thresh_attack": len(thresh_a_l),
        "test_normal": len(test_n_l),
        "test_attack": len(test_a_l),
    }
    return HoldoutSplit(
        train_idx=normal_pos[train_l],
        thresh_idx=np.concatenate([normal_pos[thresh_n_l], attack_pos[thresh_a_l]]),
        test_idx=np.concatenate([normal_pos[test_n_l], attack_pos[test_a_l]]),
        counts=counts,
        matches_notebook=counts == REFERENCE_SPLIT,
    )


def stratified_positions(positions: np.ndarray, y_all: np.ndarray, n: int | None, seed: int = SEED) -> np.ndarray:
    """Pick ``n`` positions keeping the class ratio of ``positions`` (all of them if n is None)."""
    if n is None or n >= len(positions):
        return positions
    rng = np.random.default_rng(seed)
    y = y_all[positions]
    picked = []
    for cls in (0, 1):
        pool = positions[y == cls]
        k = int(round(n * len(pool) / len(positions)))
        k = min(max(k, 1 if len(pool) else 0), len(pool))
        picked.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(picked))
