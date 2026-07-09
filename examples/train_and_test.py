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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SEAGAN on Data/Train and evaluate on Data/Test."
    )
    parser.add_argument("--data-dir", default=default_data_dir(), help="Path to Data folder.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument(
        "--output",
        default=Path("outputs") / "seagan_example_checkpoint.pt",
        help="Checkpoint path to write after training.",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_points, train_params = load_repo_split("Train", args.data_dir)
    test_points, test_params = load_repo_split("Test", args.data_dir)

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

    train_graphs, val_graphs = split_graphs(
        graphs,
        train_frac=args.train_frac,
        seed=args.seed,
    )

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
        class_weights, class_counts = compute_inverse_frequency_class_weights(
            train_graphs,
            n_classes=n_classes,
        )

    train_loader, val_loader = make_loaders(
        train_graphs,
        val_graphs,
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    model = SEAGAN(
        in_dim=train_graphs[0].x.shape[1],
        n_classes=n_classes,
        heads=args.heads,
        dropout=args.dropout,
        hidden_sizes=args.hidden_sizes,
        edge_dim=train_graphs[0].edge_attr.shape[1],
    )

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

    print(f"Class map: {class_map}")
    if class_counts is not None:
        print(f"Class counts: {class_counts.tolist()}")
        print(f"Class weights: {class_weights.tolist()}")
    print(f"Train metrics: {train_metrics}")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()

