"""Streamlit-aware glue: cached loading, cached scoring and the shared dataset selector."""
from __future__ import annotations

import streamlit as st

from . import scoring
from .artifacts import Artifacts, artifact_signature, load_artifacts
from .config import DEFAULT_DATASET
from .data import Dataset, DatasetError, HoldoutSplit, load_dataset, notebook_split


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models…")
def _cached_artifacts(signature: tuple) -> Artifacts:  # `signature` is part of the cache key
    return load_artifacts()


def get_artifacts() -> Artifacts:
    """Loaded once per server process; reloads automatically if a file in models/ changes."""
    return _cached_artifacts(artifact_signature())


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=4)
def _cached_scores(X, signature: tuple):  # `signature` is part of the cache key
    return scoring.score_dataframe(get_artifacts(), X)


def score_frame(X):
    """Score a feature frame with every available model (cached on the frame's contents)."""
    n_models = len(get_artifacts().available_models())
    with st.spinner(f"Scoring {len(X):,} flows with {n_models} model{'s' if n_models != 1 else ''}…"):
        return _cached_scores(X, artifact_signature())


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Reading the dataset (large files take a few seconds)…")
def _cached_default_dataset(path_sig: tuple, features: tuple) -> Dataset:  # args are the cache key
    return load_dataset(DEFAULT_DATASET, list(features), DEFAULT_DATASET.name)


def default_dataset(features: list[str]) -> Dataset | None:
    """The bundled CSV, or None if it is not there. Raises DatasetError if it is unreadable."""
    if not DEFAULT_DATASET.exists():
        return None
    stat = DEFAULT_DATASET.stat()
    return _cached_default_dataset((str(DEFAULT_DATASET), stat.st_mtime_ns, stat.st_size), tuple(features))


@st.cache_resource(show_spinner="Recreating the notebook's train / threshold / test split…")
def _cached_split(key: str, _ds: Dataset) -> HoldoutSplit:  # `_ds` is deliberately not hashed
    return notebook_split(_ds)


def get_split(ds: Dataset) -> HoldoutSplit:
    return _cached_split(ds.key, ds)


def dataset_selector(art: Artifacts) -> Dataset | None:
    """Sidebar control shared by every data page. Returns the active dataset or None.

    An uploaded file is parsed once and stored in ``st.session_state`` so it stays active
    when you move between pages (widget state itself is dropped on page changes).
    """
    ss = st.session_state
    with st.sidebar:
        st.markdown("#### Data")
        upload = st.file_uploader(
            "Upload a flow CSV", type=["csv"], key="upload_csv",
            help="Needs the 51 CICIDS2017 feature columns. An Attack_Binary column (0/1) is optional.",
        )

        if upload is not None:
            fingerprint = f"{upload.name}:{upload.size}"
            if fingerprint != ss.get("upload_fp") and fingerprint != ss.get("upload_dismissed"):
                try:
                    with st.spinner(f"Reading {upload.name}…"):
                        ss["upload_ds"] = load_dataset(upload, art.features, upload.name)
                    ss["upload_fp"], ss["upload_error"] = fingerprint, None
                except DatasetError as exc:
                    ss["upload_ds"], ss["upload_fp"] = None, fingerprint
                    ss["upload_error"] = str(exc)

        if ss.get("upload_error"):
            st.error(ss["upload_error"])

        active: Dataset | None = ss.get("upload_ds")
        if active is not None:
            st.success(f"Using **{active.name}** ({len(active.df):,} flows)")
            if st.button("Use the default dataset instead", width="stretch"):
                ss["upload_dismissed"] = ss.get("upload_fp")
                for k in ("upload_ds", "upload_fp", "upload_error"):
                    ss.pop(k, None)
                st.rerun()
            return active

    # Fall back to the bundled dataset.
    try:
        ds = default_dataset(art.features)
    except DatasetError as exc:
        st.sidebar.error(f"The default dataset could not be read: {exc}")
        return None
    if ds is not None:
        st.sidebar.caption(f"Using the default dataset: **{ds.name}** ({len(ds.df):,} flows).")
    return ds
