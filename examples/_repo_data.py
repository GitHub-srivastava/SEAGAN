from __future__ import annotations

from pathlib import Path

import pandas as pd

# The repository has files for 8-point through 15-point A-Ci curves.
CURVE_TYPES = tuple(range(8, 16))


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "Data"


def load_repo_split(
    split: str,
    data_dir: str | Path | None = None,
    curve_types: tuple[int, ...] = CURVE_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load this repository's Train/Test Excel files.

    The PyPI helper follows the original notebook's shuffled point-file naming.
    This repository stores point files as ACi_points_*points.xlsx, so this thin
    loader handles only the local filenames and leaves graph construction to
    seagan.build_graphs_from_df.
    """

    root = Path(data_dir) if data_dir is not None else default_data_dir()
    folder = root / split
    if not folder.exists():
        raise FileNotFoundError(f"Missing data split folder: {folder}")

    points = _load_many(folder, "ACi_points", curve_types)
    params = _load_many(folder, "ACi_params", curve_types)
    return points, params


def _load_many(folder: Path, prefix: str, curve_types: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for n_points in curve_types:
        path = folder / f"{prefix}_{n_points}points.xlsx"
        if not path.exists() and prefix == "ACi_points":
            path = folder / f"{prefix}_{n_points}points_shuffled.xlsx"

        if not path.exists():
            raise FileNotFoundError(f"Missing expected data file: {path}")

        df = pd.read_excel(path, engine="openpyxl")
        df["n_points"] = n_points
        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
