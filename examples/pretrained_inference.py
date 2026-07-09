from __future__ import annotations

import argparse

import pandas as pd

from seagan import (
    build_graphs_from_df,
    load_pretrained_seagan,
    predict_graph,
    standardize_graphs_from_checkpoint,
)

from _repo_data import default_data_dir, load_repo_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run packaged pretrained SEAGAN on one curve from this repo."
    )
    parser.add_argument("--data-dir", default=default_data_dir(), help="Path to Data folder.")
    parser.add_argument("--split", default="Test", choices=["Train", "Test"])
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_points, df_params = load_repo_split(args.split, args.data_dir)
    graphs, class_map = build_graphs_from_df(df_points, df_params)

    if not graphs:
        raise ValueError("No graphs were built from the selected data split.")
    if args.curve_index < 0 or args.curve_index >= len(graphs):
        raise IndexError(f"--curve-index must be between 0 and {len(graphs) - 1}")

    model, checkpoint = load_pretrained_seagan(args.checkpoint)
    graphs = standardize_graphs_from_checkpoint(graphs, checkpoint)

    graph = graphs[args.curve_index]
    pred_labels = predict_graph(model, graph, one_indexed=True).numpy()
    probabilities = predict_graph(model, graph, return_probabilities=True).numpy()

    points_by_curve = df_points.set_index("curve_id")
    curve_ids = list(points_by_curve.index.unique())
    curve_id = curve_ids[args.curve_index]
    curve_points = points_by_curve.loc[curve_id].reset_index()
    if isinstance(curve_points, pd.Series):
        curve_points = curve_points.to_frame().T

    output = curve_points[["curve_id", "point_id", "Ci", "Anet", "ID"]].copy()
    output = output.rename(columns={"ID": "true_label"})
    output["pred_label"] = pred_labels
    output["prob_class_1"] = probabilities[:, 0]
    output["prob_class_2"] = probabilities[:, 1]
    output["prob_class_3"] = probabilities[:, 2]

    print(f"Loaded class map: {class_map}")
    print(f"Curve id: {curve_id}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

