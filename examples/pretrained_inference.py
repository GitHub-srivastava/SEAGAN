from __future__ import annotations

import argparse
from pathlib import Path

import torch  # Import before pandas to avoid Windows DLL load-order issues in some envs.
import pandas as pd

from seagan import (
    SEAGAN,
    build_graphs_from_df,
    load_checkpoint,
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
CURVE_NUMBER_TO_CHECK = 100

# Leave this as None to use the sample checkpoint packaged with pip install seagan.
# To use a checkpoint you trained with examples/train_and_test.py, set:
# CHECKPOINT_PATH = Path("outputs") / "seagan_example_checkpoint.pt"
CHECKPOINT_PATH = None # Path("outputs") / "seagan_example_checkpoint1.pt" # 
USE_PACKAGED_CHECKPOINT = False

# Leave these as None when the checkpoint was made by examples/train_and_test.py.
# That checkpoint already stores the architecture needed to rebuild the model.
MODEL_HEADS = None
MODEL_HIDDEN_SIZES = None
MODEL_DROPOUT = None
MODEL_IN_DIM = None
MODEL_N_CLASSES = None
MODEL_EDGE_DIM = None

# This must match the graph-building setting used during training.
K_NEIGHBORS = 4

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
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-path",
        dest="checkpoint",
        default=CHECKPOINT_PATH,
        help=(
            "Optional path to a checkpoint trained with examples/train_and_test.py. "
            "If omitted, the packaged seagan pretrained checkpoint is used."
        ),
    )
    parser.add_argument(
        "--use-packaged-checkpoint",
        action="store_true",
        default=USE_PACKAGED_CHECKPOINT,
        help="Ignore CHECKPOINT_PATH and use the packaged seagan pretrained checkpoint.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=MODEL_HEADS,
        help="Override the attention heads stored in the checkpoint.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=MODEL_HIDDEN_SIZES,
        help="Override hidden layer sizes, for example: --hidden-sizes 64 64.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=MODEL_DROPOUT,
        help="Override the dropout stored in the checkpoint.",
    )
    parser.add_argument(
        "--in-dim",
        type=int,
        default=MODEL_IN_DIM,
        help="Override model input feature dimension.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=MODEL_N_CLASSES,
        help="Override number of output classes.",
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=MODEL_EDGE_DIM,
        help="Override edge feature dimension.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=K_NEIGHBORS,
        help="Number of nearest neighbors to use when rebuilding A-Ci graphs.",
    )
    parser.add_argument(
        "--hide-probabilities",
        action="store_true",
        help="Only print true and predicted labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = None if args.use_packaged_checkpoint else resolve_checkpoint_path(args.checkpoint)
    model, checkpoint = load_model(args, checkpoint_path)
    print_model_architecture(checkpoint)

    print(f"Reading {args.split} curves from: {Path(args.data_dir).resolve()}")
    df_points, df_params = load_repo_split(args.split, args.data_dir)
    graphs, class_map = build_graphs_from_df(
        df_points,
        df_params,
        class_map=checkpoint.get("class_map"),
        k=args.k_neighbors,
    )

    if not graphs:
        raise ValueError("No graphs were built from the selected data split.")

    curve_ids = list(df_points["curve_id"].drop_duplicates())
    graph_index, curve_id = choose_curve(curve_ids, args.curve_id, args.curve_number)

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
        for class_index in range(probabilities.shape[1]):
            output[f"prob_class_{class_index + 1}"] = probabilities[:, class_index]

    print(f"\nAvailable class map: {class_map}")
    print(f"Showing curve_id {curve_id} from {args.split}.")
    print(output.to_string(index=False))


def load_model(args: argparse.Namespace, checkpoint_path: Path | None):
    """Load a model, using checkpoint architecture unless overrides are supplied."""

    if checkpoint_path is None:
        print("Loading packaged SEAGAN pretrained checkpoint...")
    else:
        print(f"Loading user checkpoint: {checkpoint_path.resolve()}")

    overrides = architecture_overrides(args)
    if not overrides:
        return load_pretrained_seagan(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint.update(overrides)
    model = SEAGAN(
        in_dim=checkpoint.get("IN_DIM", 4),
        hidden_sizes=checkpoint.get("HIDDEN_SIZES", [64, 64]),
        n_classes=checkpoint.get("NCLASS", 3),
        heads=checkpoint.get("HEADS", 5),
        dropout=checkpoint.get("DROPOUT", 0.2),
        edge_dim=checkpoint.get("EDGE_DIM", 2),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint weights do not match the selected model architecture. "
            "Use the same --heads, --hidden-sizes, --in-dim, --n-classes, and "
            "--edge-dim values that were used during training."
        ) from None
    model.eval()
    return model, checkpoint


def architecture_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.heads is not None:
        overrides["HEADS"] = args.heads
    if args.hidden_sizes is not None:
        overrides["HIDDEN_SIZES"] = list(args.hidden_sizes)
    if args.dropout is not None:
        overrides["DROPOUT"] = args.dropout
    if args.in_dim is not None:
        overrides["IN_DIM"] = args.in_dim
    if args.n_classes is not None:
        overrides["NCLASS"] = args.n_classes
    if args.edge_dim is not None:
        overrides["EDGE_DIM"] = args.edge_dim
    return overrides


def print_model_architecture(checkpoint: dict) -> None:
    print(
        "Model architecture: "
        f"in_dim={checkpoint.get('IN_DIM', 4)}, "
        f"hidden_sizes={checkpoint.get('HIDDEN_SIZES', [64, 64])}, "
        f"n_classes={checkpoint.get('NCLASS', 3)}, "
        f"heads={checkpoint.get('HEADS', 5)}, "
        f"dropout={checkpoint.get('DROPOUT', 0.2)}, "
        f"edge_dim={checkpoint.get('EDGE_DIM', 2)}"
    )


def resolve_checkpoint_path(checkpoint: str | Path | None) -> Path | None:
    """Return a validated user checkpoint path, or None for the packaged checkpoint."""

    if checkpoint is None or str(checkpoint).strip() == "":
        return None

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}. "
            "Train one with examples/train_and_test.py or omit --checkpoint to use "
            "the packaged pretrained checkpoint."
        )
    if checkpoint_path.is_dir():
        raise IsADirectoryError(f"Checkpoint path points to a folder, not a file: {checkpoint_path}")

    return checkpoint_path


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
