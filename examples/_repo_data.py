from __future__ import annotations

from pathlib import Path

import pandas as pd

POINTS_FILE = "ACi_points.xlsx"
PARAMS_FILE = "ACi_params.xlsx"


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "Data"


def load_repo_split(
    split: str,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load this repository's Train/Test Excel files.

    Expected layout:
      Data/Train/ACi_points.xlsx
      Data/Train/ACi_params.xlsx
      Data/Test/ACi_points.xlsx
      Data/Test/ACi_params.xlsx
    """

    root = Path(data_dir) if data_dir is not None else default_data_dir()
    folder = root / split
    if not folder.exists():
        raise FileNotFoundError(f"Missing data split folder: {folder}")

    points = _read_excel(folder / POINTS_FILE)
    params = _read_excel(folder / PARAMS_FILE)

    return points, params


def _read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Each split folder should contain "
            f"{POINTS_FILE} and {PARAMS_FILE}."
        )

    return pd.read_excel(path, engine="openpyxl")
