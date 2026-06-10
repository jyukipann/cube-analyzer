"""Diffusion solver training (MLX).

Usage
-----
    uv run python source/train.py                 # small config, defaults
    uv run python source/train.py --d-model 256 --n-layers 6 --n-heads 8
    uv run python source/train.py --steps 50000 --save weights.npz
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cube"))

from data import generate_batch          # noqa: E402
from model.solver import CubeSolver      # noqa: E402


def loss_fn(model: CubeSolver, batch: dict) -> mx.array:
    logits = model(
        goal=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        curr=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
        t=batch['t'],
    )
    return mx.mean(nn.losses.cross_entropy(logits, batch['target']))


def accuracy(model: CubeSolver, batch: dict) -> float:
    logits = model(
        goal=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        curr=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
        t=batch['t'],
    )
    preds = mx.argmax(logits, axis=1)
    return float(mx.mean(preds == batch['target']))


def train(args: argparse.Namespace) -> None:
    model = CubeSolver(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        t_max=args.t_max,
    )
    mx.eval(model.parameters())
    print(f"Parameters: {model.n_params():,}")
    print(f"Config: d={args.d_model}, layers={args.n_layers}, "
          f"heads={args.n_heads}, batch={args.batch_size}, t_max={args.t_max}")

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = generate_batch(args.batch_size, t_max=args.t_max)
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % args.log_every == 0:
            val_batch = generate_batch(512, t_max=args.t_max)
            acc = accuracy(model, val_batch)
            mx.eval()
            elapsed = time.time() - t0
            print(f"step {step:6d}  loss {float(loss):.4f}  "
                  f"acc {acc:.3f}  "
                  f"({elapsed:.1f}s, {step / elapsed:.1f} steps/s)")

    if args.save:
        weights = dict(tree_flatten(model.parameters()))
        mx.savez(args.save, **weights)
        print(f"Saved → {args.save}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model",    type=int,   default=128)
    parser.add_argument("--n-layers",   type=int,   default=4)
    parser.add_argument("--n-heads",    type=int,   default=4)
    parser.add_argument("--t-max",      type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--steps",      type=int,   default=10_000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--log-every",  type=int,   default=100)
    parser.add_argument("--save",       type=str,   default="")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
