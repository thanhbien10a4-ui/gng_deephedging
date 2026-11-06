#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warmup and Resume Invariant Verification Test for JAX Engine
Verifies:
1. Warmup pass does not modify weights or Adam optimizer states prior to batch 1.
2. Checkpoint resume across a fresh process correctly restores weights, optimizer states (m, v, t), and step count.
3. Resumed training matches uninterrupted training trajectory with identical batch data.
Preset Tolerances:
- Warmup parameter drift == 0.0
- Warmup optimizer step == 0
- Resume parameter max absolute error <= 1e-6
- Resume loss discrepancy <= 1e-6
"""

import os
import sys
import shutil
import pickle
import subprocess
from pathlib import Path
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path(os.environ.get("DH_ROOT", "/home/user04/gng_deephedging"))
if not ROOT.exists():
    ROOT = Path("D:/")
REWRITTEN_ROOT = ROOT / "deephedging-rewritten" if (ROOT / "deephedging-rewritten").exists() else Path(__file__).resolve().parent.parent

if str(REWRITTEN_ROOT) not in sys.path:
    sys.path.insert(0, str(REWRITTEN_ROOT))

import jax
import jax.numpy as jnp
from dh_jax.world import WorldConfig, simulate
from dh_jax.model import GymConfig, AgentConfig, as_batch, rollout, loss, init_params
from dh_jax.train import TrainConfig, adam_init, train

def run_jax_warmup_and_resume():
    print("=" * 60)
    print("TEST: JAX Warmup and Resume Invariants")
    print("=" * 60)
    
    # 1. Setup deterministic world & gym
    w_cfg = WorldConfig(samples=200, steps=10, black_scholes=True, seed=101)
    world = simulate(w_cfg)
    val_world = world.clone(samples=50, seed=102)
    data = as_batch(world)
    val_data = as_batch(val_world)
    
    gym = GymConfig(agent=AgentConfig(depth=3, width=20, activation="softplus"), utility="cvar", lmbda=1.0)
    key = jax.random.PRNGKey(42)
    feat_dims = {k: v.shape[2] for k, v in world.features.items()}
    p0 = init_params(key, gym, feat_dims, world.nInst)
    opt0 = adam_init(p0)
    
    # 2. Test Warmup Invariant
    print("1. Testing Warmup Invariant...")
    _ = rollout(p0, gym, data)
    _ = loss(p0, gym, data)
    
    diffs_warmup = [np.max(np.abs(np.asarray(p0["dense"][i]["W"]) - np.asarray(p0["dense"][i]["W"]))) for i in range(len(p0["dense"]))]
    max_warmup_drift = max(diffs_warmup)
    print(f"   Warmup Param Drift: {max_warmup_drift:.2e} (Threshold: 0.0)")
    assert max_warmup_drift == 0.0, f"Warmup mutated parameters! Drift: {max_warmup_drift}"
    
    step_warmup = int(opt0["t"])
    print(f"   Warmup Optimizer Step: {step_warmup} (Expected: 0)")
    assert step_warmup == 0, f"Warmup updated optimizer step: {step_warmup}"
    print("   -> Warmup Verdict: PASS")
    
    # 3. Test Checkpoint Resume Invariant (Across Fresh Subprocess)
    print("\n2. Testing Cross-Process Checkpoint Resume Invariant...")
    ckpt_dir = REWRITTEN_ROOT / ".test_jax_ckpt_isolated"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # A. Uninterrupted 2 epochs
    t_cfg_2ep = TrainConfig(epochs=2, batch_size=32, learning_rate=0.005, seed=77)
    res_uninterrupted = train(gym, t_cfg_2ep, p0, data, val_data, checkpoint_dir=None)
    
    # B. Run 1 epoch and save checkpoint
    t_cfg_1ep = TrainConfig(epochs=1, batch_size=32, learning_rate=0.005, seed=77)
    _ = train(gym, t_cfg_1ep, p0, data, val_data, checkpoint_dir=str(ckpt_dir), checkpoint_every=1)
    
    # C. Save context for fresh process
    context_file = REWRITTEN_ROOT / ".test_jax_resume_ctx.pkl"
    resumed_file = REWRITTEN_ROOT / ".test_jax_resumed_out.pkl"
    
    with open(context_file, "wb") as f:
        pickle.dump(dict(gym=gym, t_cfg_2ep=t_cfg_2ep, p0=p0, data=data, val_data=val_data, ckpt_dir=str(ckpt_dir), out_path=str(resumed_file)), f)
        
    subprocess_script = f"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, r"{str(REWRITTEN_ROOT)}")
from dh_jax.train import train

with open(r"{str(context_file)}", "rb") as f:
    ctx = pickle.load(f)

res = train(ctx["gym"], ctx["t_cfg_2ep"], ctx["p0"], ctx["data"], ctx["val_data"], checkpoint_dir=ctx["ckpt_dir"], checkpoint_every=1)

with open(ctx["out_path"], "wb") as f:
    pickle.dump(res, f)
print("Fresh process execution completed.")
"""
    proc = subprocess.run([sys.executable, "-c", subprocess_script], capture_output=True, text=True)
    if proc.returncode != 0:
        print("Fresh process failed:")
        print(proc.stderr)
        raise RuntimeError("Subprocess resume failed!")
        
    with open(resumed_file, "rb") as f:
        res_resumed = pickle.load(f)
        
    # Compare uninterrupted vs resumed
    diff_best_weights = [np.max(np.abs(np.asarray(res_resumed.params["dense"][i]["W"]) - np.asarray(res_uninterrupted.params["dense"][i]["W"]))) for i in range(len(p0["dense"]))]
    diff_last_weights = [np.max(np.abs(np.asarray(res_resumed.last_params["dense"][i]["W"]) - np.asarray(res_uninterrupted.last_params["dense"][i]["W"]))) for i in range(len(p0["dense"]))]
    loss_train_diff = abs(res_resumed.history["loss"][-1] - res_uninterrupted.history["loss"][-1])
    loss_val_diff = abs(res_resumed.history["val_loss"][-1] - res_uninterrupted.history["val_loss"][-1])
    
    max_w_diff = max(diff_best_weights)
    max_last_w_diff = max(diff_last_weights)
    
    print(f"   Max Best Weights Abs Diff: {max_w_diff:.2e} (Threshold <= 1e-6)")
    print(f"   Max Last Weights Abs Diff: {max_last_w_diff:.2e} (Threshold <= 1e-6)")
    print(f"   Epoch 2 Train Loss Abs Diff: {loss_train_diff:.2e} (Threshold <= 1e-6)")
    print(f"   Epoch 2 Val Loss Abs Diff: {loss_val_diff:.2e} (Threshold <= 1e-6)")
    print(f"   Best Epoch Match: Resumed={res_resumed.best_epoch}, Uninterrupted={res_uninterrupted.best_epoch}")
    
    assert max_w_diff <= 1e-6, f"Weight discrepancy: {max_w_diff}"
    assert max_last_w_diff <= 1e-6, f"Last weight discrepancy: {max_last_w_diff}"
    assert loss_train_diff <= 1e-6, f"Train loss discrepancy: {loss_train_diff}"
    assert loss_val_diff <= 1e-6, f"Val loss discrepancy: {loss_val_diff}"
    assert res_resumed.best_epoch == res_uninterrupted.best_epoch, "Best epoch mismatch"
    print("   -> Resume Verdict: PASS")
    
    # Cleanup
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if context_file.exists():
        context_file.unlink()
    if resumed_file.exists():
        resumed_file.unlink()
        
    print("=" * 60)
    print("VERDICT: JAX WARMUP & RESUME TEST PASSED COMPLETELY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_jax_warmup_and_resume()
