#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical Parity Test for Rewritten TensorFlow Engine vs Original deephedging
Verifies:
1. World simulation and spot path parity
2. Rollout parity (actions, loss, gains) with identical initial weights and data
3. Initial loss parity
4. Gradient parity
5. One Adam optimization step parameter parity
Preset Tolerances:
- Rollout max absolute error <= 1e-5
- Loss relative error <= 1e-5
- One Adam step parameter max absolute error <= 1e-5
"""

import os
import sys
from pathlib import Path
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(os.environ.get("DH_ROOT", "/home/user04/gng_deephedging"))
if not ROOT.exists():
    ROOT = Path("D:/")
REWRITTEN_ROOT = ROOT / "deephedging-rewritten" if (ROOT / "deephedging-rewritten").exists() else Path(__file__).resolve().parent.parent

if str(REWRITTEN_ROOT) not in sys.path:
    sys.path.insert(0, str(REWRITTEN_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
from cdxbasics.config import Config

# Rewritten TF modules
import dh_tf.world as rw_world
import dh_tf.gym as rw_gym
import dh_tf.trainer as rw_trainer

# Original TF modules
import deephedging.world as orig_world
import deephedging.gym as orig_gym
import deephedging.trainer as orig_trainer

def run_test():
    print("=" * 60)
    print("TEST: Numerical Parity (TensorFlow Rewritten vs Original TF)")
    print("=" * 60)
    
    # 1. Config & World
    c = Config()
    c.world.samples = 200
    c.world.steps = 10
    c.world.dt = 0.02
    c.world.drift = 0.0
    c.world.cost_s = 0.0
    c.world.black_scholes = True
    c.world.seed = 9999
    
    c.gym.objective.utility = "cvar"
    c.gym.objective.lmbda = 1.0
    c.gym.agent.network.depth = 3
    c.gym.agent.network.width = 20
    c.gym.agent.network.activation = "softplus"
    
    # Build both worlds
    world_rw = rw_world.SimpleWorld_Spot_ATM(c.world)
    world_orig = orig_world.SimpleWorld_Spot_ATM(c.world)
    
    spot_diff = np.max(np.abs(world_rw.details.spot_all - world_orig.details.spot_all))
    print(f"1. Spot Path Data Max Abs Diff: {spot_diff:.2e} (Threshold: 0.0)")
    assert spot_diff == 0.0, f"Spot mismatch: {spot_diff}"
    
    # 2. Build both Gyms
    tf.random.set_seed(42)
    gym_rw = rw_gym.VanillaDeepHedgingGym(c.gym)
    gym_orig = orig_gym.VanillaDeepHedgingGym(c.gym)
    
    # Call once to build weights
    _ = gym_rw(world_rw.tf_data)
    _ = gym_orig(world_orig.tf_data)
    
    # Sync weights from gym_rw to gym_orig so weights are 100% identical
    for rw_w, orig_w in zip(gym_rw.trainable_weights, gym_orig.trainable_weights):
        orig_w.assign(rw_w)
    
    # 3. Rollout Parity
    out_rw = gym_rw(world_rw.tf_data)
    out_orig = gym_orig(world_orig.tf_data)
    
    act_diff = np.max(np.abs(out_rw["actions"].numpy() - out_orig["actions"].numpy()))
    print(f"2. Rollout Actions Max Abs Diff: {act_diff:.2e} (Threshold <= 1e-5)")
    assert act_diff <= 1e-5, f"Actions mismatch: {act_diff}"
    
    # 4. Loss Parity
    loss_rw_arr = out_rw["loss"].numpy()
    loss_orig_arr = out_orig["loss"].numpy()
    loss_path_diff = np.max(np.abs(loss_rw_arr - loss_orig_arr))
    loss_rw_mean = float(np.mean(loss_rw_arr))
    loss_orig_mean = float(np.mean(loss_orig_arr))
    rel_loss = abs(loss_rw_mean - loss_orig_mean) / max(abs(loss_orig_mean), 1e-8)
    print(f"3. Initial Loss Parity: Mean={loss_rw_mean:.8f}, MaxPathDiff={loss_path_diff:.2e}, RelDiff={rel_loss:.2e} (Threshold <= 1e-5)")
    assert rel_loss <= 1e-5, f"Loss mismatch: {rel_loss}"
    assert loss_path_diff <= 1e-5, f"Path loss mismatch: {loss_path_diff}"
    
    # 5. Gradient & 1 Adam Step Parity
    opt_rw = tf.keras.optimizers.Adam(learning_rate=0.005)
    opt_orig = tf.keras.optimizers.Adam(learning_rate=0.005)
    
    with tf.GradientTape() as tape_rw:
        l_rw = tf.reduce_mean(gym_rw(world_rw.tf_data)["loss"])
    grads_rw = tape_rw.gradient(l_rw, gym_rw.trainable_weights)
    opt_rw.apply_gradients(zip(grads_rw, gym_rw.trainable_weights))
    
    with tf.GradientTape() as tape_orig:
        l_orig = tf.reduce_mean(gym_orig(world_orig.tf_data)["loss"])
    grads_orig = tape_orig.gradient(l_orig, gym_orig.trainable_weights)
    opt_orig.apply_gradients(zip(grads_orig, gym_orig.trainable_weights))
    
    grad_diffs = [np.max(np.abs(g_rw.numpy() - g_orig.numpy())) for g_rw, g_orig in zip(grads_rw, grads_orig) if g_rw is not None]
    max_grad_diff = max(grad_diffs)
    print(f"4. Gradients Max Abs Diff: {max_grad_diff:.2e} (Threshold <= 1e-5)")
    assert max_grad_diff <= 1e-5, f"Gradient mismatch: {max_grad_diff}"
    
    param_diffs = [np.max(np.abs(w_rw.numpy() - w_orig.numpy())) for w_rw, w_orig in zip(gym_rw.trainable_weights, gym_orig.trainable_weights)]
    max_param_diff = max(param_diffs)
    print(f"5. One Adam Step Parameter Max Abs Diff: {max_param_diff:.2e} (Threshold <= 1e-5)")
    assert max_param_diff <= 1e-5, f"Parameter mismatch: {max_param_diff}"
    
    print("=" * 60)
    print("VERDICT: TENSORFLOW NUMERICAL PARITY TEST PASSED COMPLETELY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_test()
