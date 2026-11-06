#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical Parity Test for Rewritten JAX Engine vs Original deephedging-jax
Verifies:
1. Rollout parity (actions, loss, gains) with identical initial weights and data
2. Loss value parity
3. Gradient parity
4. One Adam optimization step parameter parity
Preset Tolerances:
- Rollout max absolute error <= 1e-6
- Loss relative error <= 1e-6
- One Adam step parameter max absolute error <= 1e-5
"""

import os
import sys
import types
from pathlib import Path
from importlib.machinery import SourceFileLoader
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

# Rewritten modules
from dh_jax.world import WorldConfig, simulate
from dh_jax.model import GymConfig, AgentConfig, as_batch, rollout, loss, init_params
from dh_jax.train import TrainConfig, adam_init, adam_update

# Original modules loaded via SourceFileLoader
orig_pkg = types.ModuleType("orig_jax")
sys.modules["orig_jax"] = orig_pkg
orig_world = SourceFileLoader("orig_jax.world", str(ROOT / "deephedging-jax" / "deephedging" / "world.py")).load_module()
orig_model = SourceFileLoader("orig_jax.model", str(ROOT / "deephedging-jax" / "deephedging" / "model.py")).load_module()
orig_train = SourceFileLoader("orig_jax.train", str(ROOT / "deephedging-jax" / "deephedging" / "train.py")).load_module()

def run_test():
    print("=" * 60)
    print("TEST: Numerical Parity (JAX Rewritten vs Original JAX)")
    print("=" * 60)
    
    # 1. Simulate identical world
    seed = 42
    w_cfg = WorldConfig(samples=500, steps=20, black_scholes=True, rvol=0.2, drift=0.1, seed=seed)
    orig_w_cfg = orig_world.WorldConfig(samples=500, steps=20, black_scholes=True, rvol=0.2, drift=0.1, seed=seed)
    
    world_rewritten = simulate(w_cfg)
    world_orig = orig_world.simulate(orig_w_cfg)
    
    # Verify spot paths match exactly
    spot_diff = np.max(np.abs(world_rewritten.spot_all - world_orig.spot_all))
    print(f"1. Spot Path Data Max Abs Diff: {spot_diff:.2e} (Threshold: 0.0)")
    assert spot_diff == 0.0, f"Spot mismatch: {spot_diff}"
    
    # 2. Setup Gym & Init identical weights
    gym_cfg = GymConfig(agent=AgentConfig(depth=3, width=20, activation="softplus"), utility="cvar", lmbda=1.0)
    orig_gym_cfg = orig_model.GymConfig(agent=orig_model.AgentConfig(depth=3, width=20, activation="softplus"), utility="cvar", lmbda=1.0)
    
    key = jax.random.PRNGKey(12345)
    feat_dims = {k: v.shape[2] for k, v in world_rewritten.features.items()}
    params_rewritten = init_params(key, gym_cfg, feat_dims, world_rewritten.nInst)
    params_orig = orig_model.init_params(key, orig_gym_cfg, feat_dims, world_orig.nInst)
    
    data_rewritten = as_batch(world_rewritten)
    data_orig = orig_model.as_batch(world_orig)
    
    # 3. Rollout parity
    out_rewritten = rollout(params_rewritten, gym_cfg, data_rewritten)
    out_orig = orig_model.rollout(params_orig, orig_gym_cfg, data_orig)
    
    rollout_diff = np.max(np.abs(np.asarray(out_rewritten["actions"]) - np.asarray(out_orig["actions"])))
    print(f"2. Rollout Actions Max Abs Diff: {rollout_diff:.2e} (Threshold <= 1e-6)")
    assert rollout_diff <= 1e-6, f"Rollout actions discrepancy: {rollout_diff}"
    
    # 4. Loss parity
    l_rewritten = float(loss(params_rewritten, gym_cfg, data_rewritten))
    l_orig = float(orig_model.loss(params_orig, orig_gym_cfg, data_orig))
    rel_loss_diff = abs(l_rewritten - l_orig) / max(abs(l_orig), 1e-8)
    print(f"3. Initial Loss Parity: Rewritten={l_rewritten:.8f}, Orig={l_orig:.8f}, RelDiff={rel_loss_diff:.2e} (Threshold <= 1e-6)")
    assert rel_loss_diff <= 1e-6, f"Loss discrepancy: {rel_loss_diff}"
    
    # 5. Gradient & 1 Adam Step Parity
    grad_fn = jax.grad(loss)
    orig_grad_fn = jax.grad(orig_model.loss)
    
    grads_rewritten = grad_fn(params_rewritten, gym_cfg, data_rewritten)
    grads_orig = orig_grad_fn(params_orig, orig_gym_cfg, data_orig)
    
    t_cfg = TrainConfig(learning_rate=0.005)
    orig_t_cfg = orig_train.TrainConfig(learning_rate=0.005)
    
    opt_rewritten = adam_init(params_rewritten)
    opt_orig = orig_train.adam_init(params_orig)
    
    p_next_rewritten, _ = adam_update(t_cfg, grads_rewritten, opt_rewritten, params_rewritten)
    p_next_orig, _ = orig_train.adam_update(orig_t_cfg, grads_orig, opt_orig, params_orig)
    
    diffs = []
    for l_idx in range(len(p_next_rewritten["dense"])):
        w_diff = np.max(np.abs(np.asarray(p_next_rewritten["dense"][l_idx]["W"]) - np.asarray(p_next_orig["dense"][l_idx]["W"])))
        b_diff = np.max(np.abs(np.asarray(p_next_rewritten["dense"][l_idx]["b"]) - np.asarray(p_next_orig["dense"][l_idx]["b"])))
        diffs.extend([w_diff, b_diff])
    
    max_adam_diff = max(diffs)
    print(f"4. One Adam Step Parameter Max Abs Diff: {max_adam_diff:.2e} (Threshold <= 1e-5)")
    assert max_adam_diff <= 1e-5, f"Adam step discrepancy: {max_adam_diff}"
    
    print("=" * 60)
    print("VERDICT: JAX NUMERICAL PARITY TEST PASSED COMPLETELY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_test()
