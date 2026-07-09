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

    The examples prefer the combined files:
      ACi_points_train.xlsx / ACi_params_train.xlsx
      ACi_points_test.xlsx  / ACi_params_test.xlsx

    If those files are not present, the loader falls back to the older separate
    files, such as ACi_points_8points.xlsx and ACi_params_8points.xlsx.
    """

    root = Path(data_dir) if data_dir is not None else default_data_dir()
    folder = root / split
    if not folder.exists():
        raise FileNotFoundError(f"Missing data split folder: {folder}")

    split_name = split.lower()
    points = _load_one_or_many(
        folder,
        combined_name=f"ACi_points_{split_name}.xlsx",
        prefix="ACi_points",
        curve_types=curve_types,
    )
    params = _load_one_or_many(
        folder,
        combined_name=f"ACi_params_{split_name}.xlsx",
        prefix="ACi_params",
        curve_types=curve_types,
    )
    return points, params


def _load_one_or_many(
    folder: Path,
    combined_name: str,
    prefix: str,
    curve_types: tuple[int, ...],
) -> pd.DataFrame:
    combined_path = folder / combined_name
    if combined_path.exists():
        return pd.read_excel(combined_path, engine="openpyxl")

    return _load_many(folder, prefix, curve_types)


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
