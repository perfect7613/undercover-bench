"""Resume only from real model+optimizer checkpoints, through the shared driver.

Use --checkpoint-refs <previous-group/update_result.json> with a fresh --out and
new sampling seed. Step numbers alone never restore model weights.
"""
from experiments.run_live_gpu_training import main

if __name__ == "__main__":
    main()
