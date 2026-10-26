"""Compatibility entry point for the real driver.

PR #1's synthetic token/loss/gradient generator has been removed. CPU fixtures
belong in tests and cannot claim optimizer steps or research checkpoints.
"""
from experiments.run_live_gpu_training import main

if __name__ == "__main__":
    main()
