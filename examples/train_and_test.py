from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from seagan import (
    SEAGAN,
    apply_edge_standardizer,
    apply_node_standardizer,
    build_graphs_from_df,
    compute_inverse_frequency_class_weights,
    evaluate_loader_metrics,
    fit_edge_standardizer,
    fit_node_standardizer,
    make_loaders,
    split_graphs,
    train_only,
)

from _repo_data import default_data_dir, load_repo_split


# =============================================================================
# Things you will most likely change
# =============================================================================
DATA_DIR = default_data_dir()

# For a quick first run, keep this small. For a real run, try 600-800 epochs.
EPOCHS = 30

BATCH_SIZE = 128
TRAIN_FRACTION = 0.75
RANDOM_SEED = 42

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 0.0
USE_CLASS_WEIGHTS = True

# These match the small SEAGAN example model from the PyPI package.
ATTENTION_HEADS = 5
HIDDEN_SIZES = [64, 64]
DROPOUT = 0.2
K_NEIGHBORS = 4

CHECKPOINT_TO_SAVE = Path("outputs") / "seagan_example_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SEAGAN on Data/Train and evaluate on Data/Test."
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to the Data folder.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--heads", type=int, default=ATTENTION_HEADS)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=HIDDEN_SIZES)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    parser.add_argument(
        "--output",
        default=CHECKPOINT_TO_SAVE,
        help="Checkpoint path to write after training.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        default=not USE_CLASS_WEIGHTS,
        help="Turn off inverse-frequency class weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("SEAGAN training example")
    print(f"Data folder: {Path(args.data_dir).resolve()}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden sizes: {args.hidden_sizes}, heads: {args.heads}")

    print("\n1) Loading Excel files...")
    train_points, train_params = load_repo_split("Train", args.data_dir)
    test_points, test_params = load_repo_split("Test", args.data_dir)

    print("2) Building A-Ci graphs with seagan.build_graphs_from_df...")
    graphs, class_map = build_graphs_from_df(
        train_points,
        train_params,
        k=args.k_neighbors,
    )
    test_graphs, _ = build_graphs_from_df(
        test_points,
        test_params,
        class_map=class_map,
        k=args.k_neighbors,
    )

    print("3) Splitting training graphs into train and validation groups...")
    train_graphs, val_graphs = split_graphs(
        graphs,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    print("4) Standardizing node and edge features using training graphs only...")
    x_mean, x_std = fit_node_standardizer(train_graphs)
    train_graphs = apply_node_standardizer(train_graphs, x_mean, x_std)
    val_graphs = apply_node_standardizer(val_graphs, x_mean, x_std)
    test_graphs = apply_node_standardizer(test_graphs, x_mean, x_std)

    e_mean, e_std = fit_edge_standardizer(train_graphs)
    train_graphs = apply_edge_standardizer(train_graphs, e_mean, e_std)
    val_graphs = apply_edge_standardizer(val_graphs, e_mean, e_std)
    test_graphs = apply_edge_standardizer(test_graphs, e_mean, e_std)

    n_classes = len(class_map)
    class_weights = None
    class_counts = None
    if not args.no_class_weights:
        print("5) Computing class weights for the node labels...")
        class_weights, class_counts = compute_inverse_frequency_class_weights(
            train_graphs,
            n_classes=n_classes,
        )
    else:
        print("5) Class weights are turned off.")

    train_loader, val_loader = make_loaders(
        train_graphs,
        val_graphs,
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    print("6) Creating the SEAGAN model...")
    model = SEAGAN(
        in_dim=train_graphs[0].x.shape[1],
        n_classes=n_classes,
        heads=args.heads,
        dropout=args.dropout,
        hidden_sizes=args.hidden_sizes,
        edge_dim=train_graphs[0].edge_attr.shape[1],
    )

    print("7) Training. This can take a while for large epoch counts...")
    model, train_metrics, val_metrics = train_only(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.weight_decay,
        seed=args.seed,
        gamma=args.gamma,
        class_weights=class_weights,
    )

    print("8) Testing once on the held-out Data/Test folder...")
    test_metrics = evaluate_loader_metrics(
        model,
        test_loader,
        class_weights=class_weights,
        gamma=args.gamma,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "IN_DIM": train_graphs[0].x.shape[1],
        "HIDDEN_SIZES": list(args.hidden_sizes),
        "NCLASS": n_classes,
        "HEADS": args.heads,
        "DROPOUT": args.dropout,
        "EDGE_DIM": train_graphs[0].edge_attr.shape[1],
        "EPOCH": args.epochs,
        "LR": args.lr,
        "WD": args.weight_decay,
        "GAMMA": args.gamma,
        "seed": args.seed,
        "class_map": class_map,
        "class_counts": class_counts.tolist() if class_counts is not None else None,
        "x_mean": x_mean,
        "x_std": x_std,
        "e_mean": e_mean,
        "e_std": e_std,
        "final_train_metrics": train_metrics,
        "final_validation_metrics": val_metrics,
        "final_test_metrics": test_metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    print("\nDone.")
    print(f"Class map: {class_map}")
    if class_counts is not None:
        print(f"Class counts: {class_counts.tolist()}")
        print(f"Class weights: {class_weights.tolist()}")
    print_metrics("Train", train_metrics)
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)
    print(f"Saved checkpoint: {output}")


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name} metrics")
    for key in ("loss", "accuracy", "precision", "recall", "f1", "fpr", "fnr"):
        print(f"  {key:>9}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
