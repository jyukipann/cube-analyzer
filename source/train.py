"""Cube solver training (MLX).

Supports two data modes:
  diffusion  — random-walk scrambles (original behaviour)
  cfop       — behavioral cloning from the CFOP teacher

Usage
-----
    uv run python source/train.py                               # diffusion, small config, defaults
    uv run python source/train.py --data cfop --pool-size 200000
    uv run python source/train.py --d-model 256 --n-layers 6 --n-heads 8
    uv run python source/train.py --steps 50000 --save weights.npz
    uv run python source/train.py --out-dir runs/exp1 --ckpt-every 1000
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cube"))

from data import generate_batch, load_cfop_pool   # noqa: E402
from model.solver import CubeSolver               # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, logfile=None) -> None:
    """Print to stdout and, if logfile is not None, append to it."""
    print(msg, flush=True)
    if logfile is not None:
        logfile.write(msg + "\n")
        logfile.flush()


def _sample_pool(pool: dict, batch_size: int, n_train: int | None = None) -> dict:
    """Sample batch_size random rows from the pool (pure-Python indexing).

    If n_train is given, sampling is restricted to the first n_train rows
    (i.e. the training split, excluding the validation holdout).
    """
    n = n_train if n_train is not None else pool['t'].shape[0]
    idx = [random.randrange(n) for _ in range(batch_size)]
    idx_arr = mx.array(idx)
    return {k: v[idx_arr] for k, v in pool.items()}


def _slice_pool(pool: dict, start: int, end: int) -> dict:
    """Return a contiguous slice [start:end] of the pool."""
    return {k: v[start:end] for k, v in pool.items()}


# ---------------------------------------------------------------------------
# Loss / accuracy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # --- output dir setup ---------------------------------------------------
    out_dir = args.out_dir.strip() if args.out_dir else ""
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    logfile = open(Path(out_dir) / "train.log", "a") if out_dir else None
    metrics_path = Path(out_dir) / "metrics.jsonl" if out_dir else None

    # --- model --------------------------------------------------------------
    model = CubeSolver(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        t_max=args.t_max,
    )
    mx.eval(model.parameters())

    # Determine ffn_mult: CubeSolver uses 4 as the default (ffn_dim = d_model * ffn_mult)
    # We pass it through as args.ffn_mult (default 4 matching the model default).
    ffn_mult = args.ffn_mult

    # Model config dict written alongside each checkpoint
    model_config = {
        "d_model":  args.d_model,
        "n_layers": args.n_layers,
        "n_heads":  args.n_heads,
        "ffn_mult": ffn_mult,
        "t_max":    args.t_max,
    }

    # --- optional resume from checkpoint ------------------------------------
    if args.resume:
        from mlx.utils import tree_unflatten
        model.update(tree_unflatten(list(mx.load(args.resume).items())))
        mx.eval(model.parameters())

    # --- banner -------------------------------------------------------------
    log(f"mode:     {args.data}", logfile)
    log(f"config:   d={args.d_model}, layers={args.n_layers}, "
        f"heads={args.n_heads}, ffn_mult={ffn_mult}, batch={args.batch_size}, t_max={args.t_max}",
        logfile)
    log(f"out-dir:  {out_dir if out_dir else '(none)'}", logfile)
    log(f"parameters: {model.n_params():,}", logfile)
    if args.resume:
        log(f"resumed from: {args.resume}", logfile)
    if args.data == "cfop":
        log(f"pool-size: {args.pool_size}, scramble-depth: {args.scramble_depth}",
            logfile)
        if getattr(args, 'diverse_pool', False):
            log(f"diverse-pool: ON  (min-depth={args.min_depth}, "
                f"max-depth={args.scramble_depth}, randomize=True)", logfile)

    # --- data source --------------------------------------------------------
    pool = None
    val_batch = None      # held-out validation set, built once at startup
    n_train_pool = None   # for cfop: number of pool rows reserved for training

    if args.data == "cfop":
        pool_cache = (
            str(Path(out_dir) / "cfop_pool.npz")
            if out_dir
            else "source/.cfop_pool.npz"
        )
        use_diverse = getattr(args, 'diverse_pool', False)
        log(f"loading/building CFOP pool ({args.pool_size} samples)...", logfile)
        pool = load_cfop_pool(
            args.pool_size,
            scramble_depth=args.scramble_depth,
            t_max=args.t_max,
            cache_path=pool_cache,
            verbose=True,
            min_depth=args.min_depth if use_diverse else None,
            randomize=use_diverse,
        )
        total_rows = pool['t'].shape[0]
        log(f"pool ready: {total_rows} samples", logfile)

        # Hold out the last 5% of pool rows for validation
        n_val = max(1, int(total_rows * 0.05))
        n_train_pool = total_rows - n_val
        val_batch = _slice_pool(pool, n_train_pool, total_rows)
        log(f"validation holdout: {n_val} rows (rows {n_train_pool}..{total_rows-1}); "
            f"training on first {n_train_pool} rows", logfile)

    else:
        # diffusion mode: generate a fixed val batch once with a distinct seed
        # Use Python's random module seeded separately so training RNG is unaffected.
        import random as _rnd
        _saved_state = _rnd.getstate()
        _rnd.seed(999983)   # distinct from training; any fixed value works
        val_batch = generate_batch(2048, t_max=args.t_max)
        _rnd.setstate(_saved_state)
        log("validation: 2048-sample held-out diffusion batch (seed=999983)", logfile)

    # --- optimizer ----------------------------------------------------------
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # --- loop ---------------------------------------------------------------
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.data == "cfop":
            batch = _sample_pool(pool, args.batch_size, n_train=n_train_pool)
        else:
            batch = generate_batch(args.batch_size, t_max=args.t_max)

        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % args.log_every == 0:
            # Training loss (from the most recent training batch)
            train_loss = float(loss)

            # Held-out validation metrics
            val_acc = accuracy(model, val_batch)
            val_loss_val = float(loss_fn(model, val_batch))
            mx.eval()
            elapsed = time.time() - t0
            steps_per_s = step / elapsed
            msg = (f"step {step:6d}  train_loss {train_loss:.4f}  "
                   f"val_loss {val_loss_val:.4f}  val_acc {val_acc:.3f}  "
                   f"({elapsed:.1f}s, {steps_per_s:.1f} steps/s)")
            log(msg, logfile)

            if metrics_path is not None:
                record = {
                    "step": step,
                    "train_loss": train_loss,
                    "loss": val_loss_val,    # keep backward-compatible key name
                    "acc": float(val_acc),   # now held-out accuracy
                    "steps_per_s": round(steps_per_s, 3),
                }
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps(record) + "\n")

        # --- periodic checkpoint --------------------------------------------
        if out_dir and args.ckpt_every > 0 and step % args.ckpt_every == 0:
            weights = dict(tree_flatten(model.parameters()))
            ckpt_path = str(Path(out_dir) / f"ckpt_{step}.npz")
            latest_path = str(Path(out_dir) / "latest.npz")
            mx.savez(ckpt_path, **weights)
            mx.savez(latest_path, **weights)
            # Write sibling JSON configs
            ckpt_cfg_path = str(Path(out_dir) / f"ckpt_{step}.json")
            latest_cfg_path = str(Path(out_dir) / "latest.json")
            for cfg_path in (ckpt_cfg_path, latest_cfg_path):
                with open(cfg_path, "w") as f:
                    json.dump(model_config, f, indent=2)
            log(f"checkpoint saved -> {ckpt_path} (+ config json)", logfile)

    # --- final save ---------------------------------------------------------
    if args.save:
        weights = dict(tree_flatten(model.parameters()))
        mx.savez(args.save, **weights)
        # Write sibling config JSON next to the explicit save path
        save_cfg_path = str(Path(args.save).with_suffix(".json"))
        with open(save_cfg_path, "w") as f:
            json.dump(model_config, f, indent=2)
        log(f"Saved -> {args.save} (+ {save_cfg_path})", logfile)

    if logfile is not None:
        logfile.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    # existing args (unchanged defaults)
    parser.add_argument("--d-model",    type=int,   default=128)
    parser.add_argument("--n-layers",   type=int,   default=4)
    parser.add_argument("--n-heads",    type=int,   default=4)
    parser.add_argument("--t-max",      type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--steps",      type=int,   default=10_000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--log-every",  type=int,   default=100)
    parser.add_argument("--save",       type=str,   default="")
    # new args
    parser.add_argument("--data",          choices=["diffusion", "cfop"],
                        default="diffusion")
    parser.add_argument("--out-dir",       type=str, default="")
    parser.add_argument("--ckpt-every",    type=int, default=0)
    parser.add_argument("--pool-size",     type=int, default=200_000)
    parser.add_argument("--scramble-depth", type=int, default=25)
    parser.add_argument("--resume",        type=str, default="",
                        help="path to a checkpoint .npz to continue training from")
    parser.add_argument("--diverse-pool",  action="store_true",
                        help="build a diverse CFOP pool with randomized solutions "
                             "and variable scramble depth (requires --data cfop)")
    parser.add_argument("--min-depth",     type=int, default=1,
                        help="minimum scramble depth when --diverse-pool is active "
                             "(default: 1)")
    parser.add_argument("--ffn-mult",      type=int, default=4,
                        help="FFN hidden dim multiplier (default: 4, matches model default)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
