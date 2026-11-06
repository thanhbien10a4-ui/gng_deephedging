#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Verification Runner for Deep Hedging Rewritten
Executes:
1. JAX Numerical Parity Test (test_parity_jax.py)
2. TensorFlow Numerical Parity Test (test_parity_tf.py)
3. JAX Warmup and Resume Test (test_warmup_resume_jax.py)
4. TensorFlow Warmup and Resume Test (test_warmup_resume_tf.py)
Saves UTF-8 execution logs to target directory.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

REWRITTEN_ROOT = Path("D:/deephedging-rewritten")
OUTPUT_DIR = REWRITTEN_ROOT / "results" / "windows_verification_20261106_165611"
LOGS_DIR = OUTPUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_JAX = r"C:\Users\Hi Windows 11 Home\AppData\Local\Programs\Python\Python313\python.exe"
PYTHON_TF = r"D:\deephedging-rewritten\.venv-tf\Scripts\python.exe"

TESTS = [
    {
        "name": "JAX Numerical Parity",
        "script": str(REWRITTEN_ROOT / "tests" / "test_parity_jax.py"),
        "python": PYTHON_JAX,
        "log": str(LOGS_DIR / "test_parity_jax.log"),
    },
    {
        "name": "TensorFlow Numerical Parity",
        "script": str(REWRITTEN_ROOT / "tests" / "test_parity_tf.py"),
        "python": PYTHON_TF,
        "log": str(LOGS_DIR / "test_parity_tf.log"),
    },
    {
        "name": "JAX Warmup and Resume",
        "script": str(REWRITTEN_ROOT / "tests" / "test_warmup_resume_jax.py"),
        "python": PYTHON_JAX,
        "log": str(LOGS_DIR / "test_warmup_resume_jax.log"),
    },
    {
        "name": "TensorFlow Warmup and Resume",
        "script": str(REWRITTEN_ROOT / "tests" / "test_warmup_resume_tf.py"),
        "python": PYTHON_TF,
        "log": str(LOGS_DIR / "test_warmup_resume_tf.log"),
    },
]

def main():
    print("=" * 70)
    print("STARTING COMPLETE VERIFICATION SUITE")
    print(f"Target Output Directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    results = {}
    
    for t in TESTS:
        name = t["name"]
        script = t["script"]
        py = t["python"]
        log_path = t["log"]
        
        print(f"\n[RUNNING] {name}...")
        t0 = time.time()
        
        proc = subprocess.run(
            [py, script],
            capture_output=True,
            text=True,
            cwd=str(REWRITTEN_ROOT)
        )
        elapsed = time.time() - t0
        
        combined_output = f"=== COMMAND: {py} {script} ===\n"
        combined_output += f"=== ELAPSED: {elapsed:.2f}s ===\n"
        combined_output += f"=== EXIT CODE: {proc.returncode} ===\n\n"
        combined_output += "--- STDOUT ---\n" + proc.stdout + "\n"
        combined_output += "--- STDERR ---\n" + proc.stderr + "\n"
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(combined_output)
            
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results[name] = {
            "status": status,
            "exit_code": proc.returncode,
            "elapsed": elapsed,
            "log": log_path
        }
        print(f"[{status}] {name} finished in {elapsed:.2f}s (Exit code: {proc.returncode})")
        print(f"  Log: {log_path}")
        
    print("\n" + "=" * 70)
    print("VERIFICATION SUITE SUMMARY:")
    for k, v in results.items():
        print(f"  - {k}: {v['status']} ({v['elapsed']:.2f}s)")
    print("=" * 70)

if __name__ == "__main__":
    main()
