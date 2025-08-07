#!/usr/bin/env python3
"""
Distributed training launcher for neural LLM.
Supports multi-GPU and multi-node training with PyTorch DDP.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def launch_distributed_training(args):
    """Launch distributed training using torchrun."""
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        f"--nnodes={args.nodes}",
    ]
    
    # Add node rank and master address for multi-node
    if args.nodes > 1:
        cmd.extend([
            f"--node_rank={args.node_rank}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
        ])
    
    # Add training script and arguments
    cmd.append(str(Path(__file__).parent / "train.py"))
    
    # Pass through training arguments
    if args.config:
        cmd.extend(["--config", args.config])
    if args.training_config:
        cmd.extend(["--training-config", args.training_config])
    if args.data_config:
        cmd.extend(["--data-config", args.data_config])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.profile:
        cmd.extend(["--profile", args.profile])
    
    print("🚀 Launching Distributed Training")
    print("=" * 40)
    print(f"🌍 Nodes: {args.nodes}")
    print(f"🔥 GPUs per node: {args.gpus}")
    print(f"📊 Total processes: {args.nodes * args.gpus}")
    if args.nodes > 1:
        print(f"🏠 Master address: {args.master_addr}:{args.master_port}")
        print(f"🔢 Node rank: {args.node_rank}")
    print("=" * 40)
    print(f"💻 Command: {' '.join(cmd)}")
    print("=" * 40)
    
    # Launch training
    try:
        result = subprocess.run(cmd, check=True)
        print("🎉 Distributed training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Distributed training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return 1


def main():
    """Main entry point for distributed training launcher."""
    parser = argparse.ArgumentParser(description="Launch distributed neural LLM training")
    
    # Distributed training parameters
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of this node")
    parser.add_argument("--master-addr", default="localhost", help="Master node address")
    parser.add_argument("--master-port", default="12355", help="Master node port")
    
    # Training configuration (passed through to train.py)
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--training-config", help="Path to training config YAML")
    parser.add_argument("--data-config", help="Path to data config YAML")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--profile", help="Hardware profile from hardware_configs.yaml")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.gpus < 1:
        print("❌ Number of GPUs must be at least 1")
        return 1
    
    if args.nodes < 1:
        print("❌ Number of nodes must be at least 1")
        return 1
    
    if args.nodes > 1 and (args.node_rank < 0 or args.node_rank >= args.nodes):
        print(f"❌ Node rank must be between 0 and {args.nodes - 1}")
        return 1
    
    # Check if torchrun is available
    try:
        subprocess.run(["torchrun", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ torchrun not found. Please install PyTorch >= 1.9.0")
        return 1
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    return launch_distributed_training(args)


if __name__ == "__main__":
    sys.exit(main())