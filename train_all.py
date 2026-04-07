#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import importlib.util


def _remove_opt_with_value(argv: list[str], *names: str) -> list[str]:
    """Remove --key value, -k value, and --key=value occurrences for the provided option names."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = False
        for name in names:
            if token == name:
                matched = True
                # drop this token and (if present) its value
                i += 2
                break
            if token.startswith(name + "="):
                matched = True
                i += 1
                break
        if not matched:
            out.append(token)
            i += 1
    return out


def _has_opt(argv: list[str], *names: str) -> bool:
    return any(tok == name or tok.startswith(name + "=") for tok in argv for name in names)


def _run(cmd: list[str], cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> None:
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        print(f"\n# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _make_env(cuda_visible_devices: str | None) -> dict[str, str] | None:
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def _torchrun_cmd(
    torchrun: str,
    script: Path,
    script_args: list[str],
    nproc_per_node: int,
    master_port: int,
) -> list[str]:
    # Support both modern `torchrun` and older launcher `python -m torch.distributed.run`.
    # If user left default 'torchrun' but it's not on PATH, fall back automatically.
    torchrun_on_path = shutil.which(torchrun) is not None
    if torchrun == "torchrun" and not torchrun_on_path:
        # Try newer module first, then fall back to legacy launcher.
        if importlib.util.find_spec("torch.distributed.run") is not None:
            base = [sys.executable, "-m", "torch.distributed.run"]
            return [
                *base,
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={master_port}",
                str(script),
                *script_args,
            ]

        # Legacy fallback (available in older PyTorch): uses env vars when --use_env is passed.
        base = [sys.executable, "-m", "torch.distributed.launch", "--use_env"]
        return [
            *base,
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={master_port}",
            str(script),
            *script_args,
        ]

    # --standalone is simplest for single-node runs.
    return [
        torchrun,
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        str(script),
        *script_args,
    ]


def _pick_checkpoint(pretrain_dir: Path, prefer: str) -> Path:
    best = pretrain_dir / "best.pth"
    latest = pretrain_dir / "latest.pth"

    if prefer == "best":
        chosen = best
    elif prefer == "latest":
        chosen = latest
    else:  # auto
        chosen = best if best.exists() else latest

    if not chosen.exists():
        raise FileNotFoundError(
            f"No checkpoint found. Expected one of: {best} or {latest}"
        )
    return chosen


def main() -> None:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Stage-1: run train.py into exp_dir/hrformer_au; "
            "Stage-2: initialize train_mt.py from stage-1 checkpoint and save into exp_dir/hrformer_au_mt."
        )
    )
    parser.add_argument(
        "--stage1_exp_dir",
        default=str(here / "exp_dir" / "hrformer_au"),
        help="Experiment dir for train.py outputs",
    )
    parser.add_argument(
        "--stage2_exp_dir",
        default=str(here / "exp_dir" / "hrformer_au_mt"),
        help="Experiment dir for train_mt.py outputs",
    )
    parser.add_argument(
        "--seed_ckpt",
        choices=["auto", "best", "latest"],
        default="auto",
        help="Which stage-1 checkpoint to use to seed stage-2",
    )

    # Multi-GPU launcher options
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated GPU ids for CUDA_VISIBLE_DEVICES (e.g. '1,2,3,4'). "
            "When using DDP, local ranks 0..N-1 map to these devices."
        ),
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Run BOTH stages using torchrun DDP (multi-GPU).",
    )
    parser.add_argument(
        "--stage1_ddp",
        action="store_true",
        help="Run stage 1 using torchrun DDP.",
    )
    parser.add_argument(
        "--stage2_ddp",
        action="store_true",
        help="Run stage 2 using torchrun DDP.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="Number of DDP processes/GPUs to use (default: len(--devices) if provided).",
    )
    parser.add_argument(
        "--torchrun",
        default="torchrun",
        help="torchrun executable (default: torchrun)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for torchrun (stage 2 uses master_port+1)",
    )

    parser.add_argument(
        "--train_args",
        default="",
        help="Extra args forwarded to train.py (string). Do not include --exp_dir/-ed.",
    )
    parser.add_argument(
        "--mt_args",
        default="",
        help="Extra args forwarded to train_mt.py (string). Do not include --exp_dir/-ed.",
    )

    parser.add_argument(
        "--hardcode_batch",
        action="store_true",
        help=(
            "When running DDP, inject a per-GPU --batch_size if you didn't set one explicitly. "
            "Stage1 defaults to 256/nproc, stage2 defaults to 16/nproc."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands/copy plan without running",
    )

    args = parser.parse_args()

    train_script = here / "train.py"
    mt_script = here / "train_mt.py"
    if not train_script.exists():
        raise SystemExit(f"Missing {train_script}")
    if not mt_script.exists():
        raise SystemExit(f"Missing {mt_script}")

    stage1_exp_dir = Path(args.stage1_exp_dir).expanduser().resolve()
    stage2_exp_dir = Path(args.stage2_exp_dir).expanduser().resolve()

    use_ddp_stage1 = bool(args.ddp or args.stage1_ddp)
    use_ddp_stage2 = bool(args.ddp or args.stage2_ddp)

    if args.nproc_per_node is not None:
        nproc = int(args.nproc_per_node)
    elif args.devices:
        nproc = len([d for d in args.devices.split(",") if d.strip()])
    else:
        nproc = 4 if (use_ddp_stage1 or use_ddp_stage2) else 1

    if (use_ddp_stage1 or use_ddp_stage2) and nproc <= 1:
        raise SystemExit("DDP requested but nproc_per_node resolves to <= 1")

    env = _make_env(args.devices)

    # Make DDP failures fail-fast instead of hanging forever.
    # Safe to set even when not using DDP.
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_BLOCKING_WAIT", "1")

    # Build argv lists from strings, and ensure exp_dir is controlled by this wrapper
    train_argv = shlex.split(args.train_args)
    train_argv = _remove_opt_with_value(train_argv, "--exp_dir", "-ed")
    train_argv += ["--exp_dir", str(stage1_exp_dir)]

    mt_argv = shlex.split(args.mt_args)
    mt_argv = _remove_opt_with_value(mt_argv, "--exp_dir", "-ed")
    mt_argv += ["--exp_dir", str(stage2_exp_dir)]

    # Hard-code per-GPU batch size for DDP runs unless user explicitly provides one.
    # This keeps global batch size consistent with the original single-GPU defaults.
    if args.hardcode_batch and use_ddp_stage1 and (not _has_opt(train_argv, "--batch_size", "-b")):
        train_bs = max(1, 256 // nproc)
        train_argv += ["--batch_size", str(train_bs)]

    if args.hardcode_batch and use_ddp_stage2 and (not _has_opt(mt_argv, "--batch_size", "-b")):
        mt_bs = max(1, 16 // nproc)
        mt_argv += ["--batch_size", str(mt_bs)]

    # Stage 1
    if use_ddp_stage1:
        cmd1 = _torchrun_cmd(args.torchrun, train_script, train_argv, nproc_per_node=nproc, master_port=int(args.master_port))
    else:
        cmd1 = [sys.executable, str(train_script)] + train_argv
    _run(cmd1, cwd=here, dry_run=args.dry_run, env=env)

    # Determine which checkpoint to seed stage 2
    stage1_pretrain = stage1_exp_dir / "pretrain"
    if args.dry_run:
        if args.seed_ckpt == "latest":
            seed_ckpt = stage1_pretrain / "latest.pth"
        elif args.seed_ckpt == "best":
            seed_ckpt = stage1_pretrain / "best.pth"
        else:
            seed_ckpt = stage1_pretrain / "best.pth"
    else:
        seed_ckpt = _pick_checkpoint(stage1_pretrain, args.seed_ckpt)

    # Prepare stage 2 checkpoint path: train_mt.py loads from <exp_dir>/pretrain/latest.pth when --resume
    stage2_pretrain = stage2_exp_dir / "pretrain"
    if not args.dry_run:
        stage2_pretrain.mkdir(parents=True, exist_ok=True)
    dst_ckpt = stage2_pretrain / "latest.pth"

    print(f"\nCopy seed checkpoint:\n  from: {seed_ckpt}\n  to:   {dst_ckpt}")
    if not args.dry_run:
        shutil.copy2(seed_ckpt, dst_ckpt)

    # Stage 2
    # NOTE: in train_mt.py, --resume also changes start_epoch via args['start_epoch'].
    # We want to *load weights* but start from epoch 0, so force --start_epoch 0.
    if not _has_opt(mt_argv, "--resume"):
        mt_argv += ["--resume"]

    if not _has_opt(mt_argv, "--start_epoch"):
        mt_argv += ["--start_epoch", "0"]

    if use_ddp_stage2:
        cmd2 = _torchrun_cmd(args.torchrun, mt_script, mt_argv, nproc_per_node=nproc, master_port=int(args.master_port) + 1)
    else:
        cmd2 = [sys.executable, str(mt_script)] + mt_argv
    _run(cmd2, cwd=here, dry_run=args.dry_run, env=env)


if __name__ == "__main__":
    main()
