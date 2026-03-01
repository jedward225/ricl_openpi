"""Train RICL on RLBench (success-only retrieval).

Convenience wrapper that defaults to pi0_fast_rlbench_ricl config.
The actual training logic is in train_pi0_fast_ricl.py.

Usage:
    cd aha_ricl

    # Default config:
    uv run scripts/train_ricl_rlbench.py --exp_name=rlbench_ricl_v1

    # Override hyperparameters:
    uv run scripts/train_ricl_rlbench.py --exp_name=rlbench_ricl_v1 \
        --batch_size=8 --num_train_steps=10000

    # Or use train_pi0_fast_ricl.py directly:
    uv run scripts/train_pi0_fast_ricl.py pi0_fast_rlbench_ricl --exp_name=rlbench_ricl_v1
"""

import sys

import openpi.training.config as _config

# Import the main training function from the RICL trainer
from train_pi0_fast_ricl import main


if __name__ == "__main__":
    # Default to pi0_fast_rlbench_ricl config if no config name is given
    if len(sys.argv) == 1 or sys.argv[1].startswith("--"):
        # Insert the config name as the first argument
        sys.argv.insert(1, "pi0_fast_rlbench_ricl")

    main(_config.cli())
