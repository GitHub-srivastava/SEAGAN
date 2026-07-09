# SEAGAN Example Data Repository

This repository contains the A-Ci curve data used to demonstrate SEAGAN from the
published PyPI package.

Install the package first:

```powershell
python -m pip install seagan
```

The examples use the `Data/Train` and `Data/Test` Excel files in this repository.
They follow the training notebook in `GNN_FC_GAT_Focal.ipynb`, but import the
model and reusable utilities from `seagan` instead of redefining them inline.

## Pretrained Use Case

Run inference with the packaged pretrained SEAGAN checkpoint:

```powershell
python examples/pretrained_inference.py
```

This builds graphs from `Data/Test`, loads the bundled checkpoint with
`load_pretrained_seagan`, applies the checkpoint standardization statistics, and
prints node-level limitation-state predictions for one curve.

Useful options:

```powershell
python examples/pretrained_inference.py --curve-index 5
python examples/pretrained_inference.py --checkpoint path\to\checkpoint.pt
```

## Training And Testing Example

Train a new SEAGAN model on `Data/Train`, validate on a split of the training
graphs, then evaluate once on `Data/Test`:

```powershell
python examples/train_and_test.py --epochs 30
```

For a notebook-like longer run, increase `--epochs` and tune the same parameters
used in the notebook:

```powershell
python examples/train_and_test.py --epochs 800 --heads 5 --hidden-sizes 64 64 --gamma 0.0
```

The script saves a checkpoint to `outputs/seagan_example_checkpoint.pt` by
default. That checkpoint includes the model state, training parameters, class
map, node standardization statistics, edge standardization statistics, and final
metrics.

