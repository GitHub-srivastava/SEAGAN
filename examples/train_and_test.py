from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from seagan import (
    SEAGAN,
    apply_edge_standardizer,
    apply_node_standardizer,
    build_graphs_from_df,
    compute_inverse_frequency_class_weights,
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
EPOCHS = 500

BATCH_SIZE = 128
TRAIN_FRACTION = 0.75
RANDOM_SEED = 42

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 0.0

# Keep this True when you want the focal-loss training step to compensate for
# class imbalance. Evaluation metrics are still unweighted class means.
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
        help="Turn off inverse-frequency class weights in the training loss.",
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
        print("5) Computing class weights for the focal training loss...")
        class_weights, class_counts = compute_inverse_frequency_class_weights(
            train_graphs,
            n_classes=n_classes,
        )
    else:
        print("5) Class weights are turned off for the training loss.")

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

    print("8) Evaluating with simple class-mean metrics...")
    train_metrics = evaluate_loader_class_mean(
        model,
        train_loader,
        n_classes=n_classes,
        gamma=args.gamma,
    )
    val_metrics = evaluate_loader_class_mean(
        model,
        val_loader,
        n_classes=n_classes,
        gamma=args.gamma,
    )
    test_metrics = evaluate_loader_class_mean(
        model,
        test_loader,
        n_classes=n_classes,
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
        "class_weights_used_for_training_loss": not args.no_class_weights,
        "class_counts": class_counts.tolist() if class_counts is not None else None,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
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


@torch.no_grad()
def evaluate_loader_class_mean(
    model,
    loader,
    n_classes: int,
    gamma: float = 0.0,
    device=None,
) -> dict[str, object]:
    """Evaluate by scoring each class first, then averaging the classes.

    This avoids a common problem with imbalanced A-Ci labels: the class with the
    most points should not dominate the final reported metrics. These metrics do
    not use the training class weights.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)
    loss_sum_by_class = torch.zeros(n_classes, dtype=torch.float64)
    count_by_class = torch.zeros(n_classes, dtype=torch.float64)

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        per_node_loss = F.cross_entropy(logits, batch.y, reduction="none")

        if gamma != 0.0:
            true_probability = torch.softmax(logits, dim=1)[
                torch.arange(batch.y.numel(), device=device),
                batch.y,
            ]
            per_node_loss = ((1.0 - true_probability) ** gamma) * per_node_loss

        predictions = logits.argmax(dim=1)

        for class_id in range(n_classes):
            class_mask = batch.y == class_id
            class_count = int(class_mask.sum().item())
            if class_count > 0:
                loss_sum_by_class[class_id] += per_node_loss[class_mask].sum().cpu()
                count_by_class[class_id] += class_count

        for true_label, pred_label in zip(batch.y.cpu(), predictions.cpu()):
            confusion[int(true_label), int(pred_label)] += 1

    per_class = {}
    for class_id in range(n_classes):
        true_positive = float(confusion[class_id, class_id])
        false_negative = float(confusion[class_id, :].sum() - confusion[class_id, class_id])
        false_positive = float(confusion[:, class_id].sum() - confusion[class_id, class_id])
        true_negative = float(confusion.sum() - true_positive - false_negative - false_positive)

        class_loss = safe_divide(
            float(loss_sum_by_class[class_id]),
            float(count_by_class[class_id]),
        )
        class_accuracy = safe_divide(true_positive, true_positive + false_negative)
        class_precision = safe_divide(true_positive, true_positive + false_positive)
        class_recall = class_accuracy
        class_f1 = safe_divide(
            2.0 * class_precision * class_recall,
            class_precision + class_recall,
        )
        class_fpr = safe_divide(false_positive, false_positive + true_negative)
        class_fnr = safe_divide(false_negative, false_negative + true_positive)

        per_class[class_id + 1] = {
            "loss": class_loss,
            "accuracy": class_accuracy,
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "fpr": class_fpr,
            "fnr": class_fnr,
            "support": int(count_by_class[class_id].item()),
        }

    return {
        "loss": mean_metric(per_class, "loss"),
        "accuracy": mean_metric(per_class, "accuracy"),
        "precision": mean_metric(per_class, "precision"),
        "recall": mean_metric(per_class, "recall"),
        "f1": mean_metric(per_class, "f1"),
        "fpr": mean_metric(per_class, "fpr"),
        "fnr": mean_metric(per_class, "fnr"),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def mean_metric(per_class: dict[int, dict[str, float]], metric_name: str) -> float:
    values = [
        class_metrics[metric_name]
        for class_metrics in per_class.values()
        if not torch.isnan(torch.tensor(class_metrics[metric_name]))
    ]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name} metrics, simple mean across classes")
    for key in ("loss", "accuracy", "precision", "recall", "f1", "fpr", "fnr"):
        print(f"  {key:>9}: {metrics[key]:.4f}")

    print("  per-class accuracy:")
    for class_id, class_metrics in metrics["per_class"].items():
        print(
            f"    class {class_id}: "
            f"{class_metrics['accuracy']:.4f} "
            f"(n={class_metrics['support']})"
        )


if __name__ == "__main__":
    main()
