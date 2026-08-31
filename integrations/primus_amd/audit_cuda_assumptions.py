#!/usr/bin/env python3
"""Static audit of CUDA/NCCL/NVTE assumptions in SkyRL Megatron path.

Runs without GPU — useful on dev machines before AMD hardware testing.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    ROOT / "skyrl/backends/skyrl_train/workers/megatron",
    ROOT / "skyrl/backends/skyrl_train/distributed/megatron",
    ROOT / "skyrl/backends/skyrl_train/weight_sync",
    ROOT / "skyrl/train/utils/utils.py",
]

PATTERNS = [
    ("torch.cuda", re.compile(r"torch\.cuda")),
    ("CUDA env", re.compile(r"CUDA_[A-Z_]+")),
    ("NVTE env", re.compile(r"NVTE_[A-Z_]+")),
    ("cuda:nccl", re.compile(r"cuda:nccl")),
    ("CUDA IPC", re.compile(r"CUDA.?IPC|cuda_ipc|reduce_tensor", re.I)),
    ("nixl", re.compile(r"\bnixl\b", re.I)),
]


def main() -> None:
    print("SkyRL Megatron CUDA/ROCm static audit\n")
    total = 0
    for base in TARGETS:
        if base.is_file():
            files = [base]
        else:
            files = sorted(base.rglob("*.py"))
        if not files:
            continue
        print(f"=== {base.relative_to(ROOT)} ===")
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(ROOT)
            for label, pat in PATTERNS:
                matches = list(pat.finditer(text))
                if matches:
                    total += len(matches)
                    lines = sorted({text.count("\n", 0, m.start()) + 1 for m in matches})
                    preview = ", ".join(str(l) for l in lines[:8])
                    suffix = "..." if len(lines) > 8 else ""
                    print(f"  {rel}: {label} ({len(matches)} hits, lines {preview}{suffix})")
        print()
    print(f"Total pattern hits: {total}")
    print("\nReview these before expecting unmodified Megatron training on ROCm.")


if __name__ == "__main__":
    main()
