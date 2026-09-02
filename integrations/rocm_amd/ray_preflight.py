#!/usr/bin/env python3
"""Verify ROCm GPUs are visible to PyTorch and Ray before GRPO."""
import os
import sys

import ray
from ray.util.placement_group import placement_group


def main() -> int:
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")

    import torch

    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")

    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        print(
            f"FAIL: need {num_gpus} visible GPUs, torch sees {torch.cuda.device_count()}"
        )
        return 1

    ray.init(
        num_gpus=num_gpus,
        include_dashboard=False,
        logging_level="error",
        log_to_driver=True,
    )
    resources = ray.cluster_resources()
    print(f"ray.cluster_resources={resources}")
    gpu_avail = resources.get("GPU", 0)
    if gpu_avail < num_gpus:
        print(f"FAIL: Ray reports GPU={gpu_avail}, need {num_gpus}")
        ray.shutdown()
        return 1

    pg = placement_group([{"GPU": 1, "CPU": 1}] * num_gpus, strategy="PACK")
    try:
        ray.get(pg.ready(), timeout=120)
        print(f"PASS: placement group ready ({num_gpus} bundles)")
    except Exception as exc:
        print(f"FAIL: placement group: {exc}")
        ray.shutdown()
        return 1

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
