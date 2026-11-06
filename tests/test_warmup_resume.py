#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warmup and Resume Invariant Verification Test
Verifies:
1. Warmup pass does not modify weights or Adam optimizer states prior to batch 1.
2. Checkpoint resume correctly restores weights, optimizer states (m, v, t), and step count.
3. Resumed training matches uninterrupted training trajectory.

Supports execution in both isolated environments:
- JAX: executes JAX warmup & resume
- TensorFlow: executes TF warmup & resume
"""

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DH_ROOT", "/home/user04/gng_deephedging"))
if not ROOT.exists():
    ROOT = Path("D:/")
REWRITTEN_ROOT = ROOT / "deephedging-rewritten" if (ROOT / "deephedging-rewritten").exists() else Path(__file__).resolve().parent.parent

if str(REWRITTEN_ROOT) not in sys.path:
    sys.path.insert(0, str(REWRITTEN_ROOT))

# Detect available frameworks
has_jax = False
try:
    import jax
    has_jax = True
except ImportError:
    pass

has_tf = False
try:
    import tensorflow as tf
    has_tf = True
except ImportError:
    pass

if __name__ == "__main__":
    if has_jax:
        from tests.test_warmup_resume_jax import run_jax_warmup_and_resume
        run_jax_warmup_and_resume()
    elif has_tf:
        from tests.test_warmup_resume_tf import run_tf_warmup_and_resume
        run_tf_warmup_and_resume()
    else:
        print("ERROR: Neither JAX nor TensorFlow is installed in current Python interpreter.")
        sys.exit(1)
