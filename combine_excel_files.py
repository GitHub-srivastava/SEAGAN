from __future__ import annotations

from pathlib import Path

import pandas as pd


# =============================================================================
# Things you may want to change
# =============================================================================
DATA_DIR = Path("Data")
SPLITS = ["Train", "Test"]
CURVE_POINT_COUNTS = range(8, 16)

POINTS_PREFIX = "ACi_points"
PARAMS_PREFIX = "ACi_params"


def main() -> None:
    for split in SPLITS:
        folder = DATA_DIR / split
        split_name = split.lower()

        points_output = folder / f"ACi_points_{split_name}.xlsx"
        params_output = folder / f"ACi_params_{split_name}.xlsx"

        points = combine_files(folder, POINTS_PREFIX)
        params = combine_files(folder, PARAMS_PREFIX)

        points.to_excel(points_output, index=False)
        params.to_excel(params_output, index=False)

        print(f"{split}: saved {len(points):,} point rows to {points_output}")
        print(f"{split}: saved {len(params):,} parameter rows to {params_output}")


def combine_files(folder: Path, prefix: str) -> pd.DataFrame:
    all_tables = []

    for n_points in CURVE_POINT_COUNTS:
        file_path = folder / f"{prefix}_{n_points}points.xlsx"
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find {file_path}")

        table = pd.read_excel(file_path, engine="openpyxl")
        table["n_points"] = n_points
        table["source_file"] = file_path.name
        all_tables.append(table)

    return pd.concat(all_tables, ignore_index=True)


if __name__ == "__main__":
    main()
