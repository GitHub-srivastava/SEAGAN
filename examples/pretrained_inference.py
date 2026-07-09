from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seagan import (
    build_graphs_from_df,
    load_pretrained_seagan,
    predict_graph,
    standardize_graphs_from_checkpoint,
)

from _repo_data import default_data_dir, load_repo_split


# =============================================================================
# Things you will most likely change
# =============================================================================
DATA_DIR = default_data_dir()
DATA_SPLIT = "Test"

# Pick the curve you want to inspect.
# Option 1: use an actual curve_id from the Excel files, for example 101.
# Option 2: leave CURVE_ID_TO_CHECK as None and use CURVE_NUMBER_TO_CHECK.
CURVE_ID_TO_CHECK = None
CURVE_NUMBER_TO_CHECK = 0

# Leave this as None to use the sample checkpoint packaged with pip install seagan.
CHECKPOINT_PATH = None

# Set this to False if you only want predicted labels.
SHOW_CLASS_PROBABILITIES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use pretrained SEAGAN on one curve from Data/Train or Data/Test."
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to the Data folder.")
    parser.add_argument("--split", default=DATA_SPLIT, choices=["Train", "Test"])
    parser.add_argument(
        "--curve-id",
        type=int,
        default=CURVE_ID_TO_CHECK,
        help="Actual curve_id to inspect. This is the most human-friendly option.",
    )
    parser.add_argument(
        "--curve-number",
        type=int,
        default=CURVE_NUMBER_TO_CHECK,
        help="Zero-based curve position to inspect when --curve-id is not given.",
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Optional checkpoint path.")
    parser.add_argument(
        "--hide-probabilities",
        action="store_true",
        help="Only print true and predicted labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading {args.split} curves from: {Path(args.data_dir).resolve()}")
    df_points, df_params = load_repo_split(args.split, args.data_dir)
    graphs, class_map = build_graphs_from_df(df_points, df_params)

    if not graphs:
        raise ValueError("No graphs were built from the selected data split.")

    curve_ids = list(df_points["curve_id"].drop_duplicates())
    graph_index, curve_id = choose_curve(curve_ids, args.curve_id, args.curve_number)

    print("Loading pretrained SEAGAN checkpoint...")
    model, checkpoint = load_pretrained_seagan(args.checkpoint)
    graphs = standardize_graphs_from_checkpoint(graphs, checkpoint)

    graph = graphs[graph_index]
    pred_labels = predict_graph(model, graph, one_indexed=True).numpy()
    show_probabilities = SHOW_CLASS_PROBABILITIES and not args.hide_probabilities
    probabilities = None
    if show_probabilities:
        probabilities = predict_graph(model, graph, return_probabilities=True).numpy()

    points_by_curve = df_points.set_index("curve_id")
    curve_points = points_by_curve.loc[curve_id].reset_index()
    if isinstance(curve_points, pd.Series):
        curve_points = curve_points.to_frame().T

    output = curve_points[["curve_id", "point_id", "Ci", "Anet", "ID"]].copy()
    output = output.rename(columns={"ID": "true_label"})
    output["pred_label"] = pred_labels
    if show_probabilities and probabilities is not None:
        output["prob_class_1"] = probabilities[:, 0]
        output["prob_class_2"] = probabilities[:, 1]
        output["prob_class_3"] = probabilities[:, 2]

    print(f"\nAvailable class map: {class_map}")
    print(f"Showing curve_id {curve_id} from {args.split}.")
    print(output.to_string(index=False))


def choose_curve(
    curve_ids: list[int],
    requested_curve_id: int | None,
    requested_curve_number: int,
) -> tuple[int, int]:
    """Return the graph position and curve_id the user asked for."""

    if requested_curve_id is not None:
        if requested_curve_id not in curve_ids:
            preview = ", ".join(str(value) for value in curve_ids[:12])
            raise ValueError(
                f"curve_id {requested_curve_id} was not found. "
                f"First available curve ids are: {preview}"
            )
        graph_index = curve_ids.index(requested_curve_id)
        return graph_index, requested_curve_id

    if requested_curve_number < 0 or requested_curve_number >= len(curve_ids):
        raise IndexError(
            f"curve number must be between 0 and {len(curve_ids) - 1}. "
            "Use --curve-id if you want to choose by the Excel curve_id value."
        )

    curve_id = curve_ids[requested_curve_number]
    return requested_curve_number, curve_id


if __name__ == "__main__":
    main()
