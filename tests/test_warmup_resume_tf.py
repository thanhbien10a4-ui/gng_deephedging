#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warmup and Resume Invariant Verification Test for TensorFlow Engine
Verifies:
1. Warmup pass does not modify weights prior to training.
2. Checkpoint/cache resume behavior across a fresh process:
   - Evaluates continuous 2 epochs vs (1 epoch -> save cache -> resume in fresh process for epoch 2).
   - Reports weight differences, Adam optimizer restoration status, and step alignment.
Preset Tolerances:
- Warmup parameter drift == 0.0
- Resume parameter max absolute error <= 1e-5
"""

import os
import sys
import shutil
import pickle
import subprocess
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

import tensorflow as tf
from cdxbasics.config import Config
import dh_tf.world as rw_world
import dh_tf.gym as rw_gym
import dh_tf.trainer as rw_trainer

def make_cfg(epochs, cache_dir, mode="on"):
    c = Config()
    c.world.samples = 100
    c.world.steps = 5
    c.world.dt = 0.02
    c.world.drift = 0.0
    c.world.cost_s = 0.0
    c.world.black_scholes = True
    c.world.seed = 42

    c.gym.objective.utility = "cvar"
    c.gym.objective.lmbda = 1.0
    c.gym.agent.network.depth = 2
    c.gym.agent.network.width = 10
    c.gym.agent.network.activation = "softplus"

    c.caching.directory = str(cache_dir)
    c.caching.mode = mode
    c.caching.epoch_freq = 1
    c.train.epochs = epochs
    c.train.tf_verbose = 0
    c.output_level = "quiet"
    return c

def run_tf_warmup_and_resume():
    print("=" * 60)
    print("TEST: TensorFlow Warmup and Resume Invariants")
    print("=" * 60)
    
    # 1. Warmup Test
    print("1. Testing Warmup Invariant...")
    c_warm = make_cfg(1, "./.cache_unused", mode="off")
    w_warm = rw_world.SimpleWorld_Spot_ATM(c_warm.world)
    g_warm = rw_gym.VanillaDeepHedgingGym(c_warm.gym)
    _ = g_warm(w_warm.tf_data)
    w_init = [w.numpy().copy() for w in g_warm.trainable_weights]
    for _ in range(3):
        _ = g_warm(w_warm.tf_data)
    w_post = [w.numpy() for w in g_warm.trainable_weights]
    warmup_diff = max([np.max(np.abs(w1 - w2)) for w1, w2 in zip(w_init, w_post)])
    print(f"   Warmup Param Drift: {warmup_diff:.2e} (Threshold: 0.0)")
    assert warmup_diff == 0.0, f"TF warmup mutated parameters! Drift: {warmup_diff}"
    print("   -> Warmup Verdict: PASS")
    
    # 2. Continuous 2 epochs
    print("\n2. Testing Cross-Process Checkpoint/Cache Resume Invariant...")
    cache_dir = REWRITTEN_ROOT / ".test_tf_cache_isolated"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    tf.keras.utils.set_random_seed(42)
    c_cont = make_cfg(2, "./.cache_unused", mode="off")
    w_cont = rw_world.SimpleWorld_Spot_ATM(c_cont.world)
    vw_cont = w_cont.clone(samples=50, seed=43)
    g_cont = rw_gym.VanillaDeepHedgingGym(c_cont.gym)
    rw_trainer.train(gym=g_cont, world=w_cont, val_world=vw_cont, config=c_cont)
    cont_weights = [w.copy() for w in g_cont.get_weights()]
    
    # 3. Epoch 1 with cache
    tf.keras.utils.set_random_seed(42)
    c_ep1 = make_cfg(1, cache_dir, mode="on")
    w_ep1 = rw_world.SimpleWorld_Spot_ATM(c_ep1.world)
    vw_ep1 = w_ep1.clone(samples=50, seed=43)
    g_ep1 = rw_gym.VanillaDeepHedgingGym(c_ep1.gym)
    rw_trainer.train(gym=g_ep1, world=w_ep1, val_world=vw_ep1, config=c_ep1)
    
    cached_files = list(cache_dir.glob("*.pck"))
    print(f"   Cache saved after epoch 1. Found {len(cached_files)} cache files.")
    assert len(cached_files) > 0, "No cache file was written after epoch 1!"
    
    # 4. Fresh Process Resume for Epoch 2
    res_weights_file = REWRITTEN_ROOT / ".test_tf_res_weights.pkl"
    subprocess_code = f"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, r"{str(REWRITTEN_ROOT)}")
import tensorflow as tf
from cdxbasics.config import Config
import dh_tf.world as rw_world
import dh_tf.gym as rw_gym
import dh_tf.trainer as rw_trainer

def make_cfg(epochs, cache_dir, mode="on"):
    c = Config()
    c.world.samples = 100
    c.world.steps = 5
    c.world.dt = 0.02
    c.world.drift = 0.0
    c.world.cost_s = 0.0
    c.world.black_scholes = True
    c.world.seed = 42

    c.gym.objective.utility = "cvar"
    c.gym.objective.lmbda = 1.0
    c.gym.agent.network.depth = 2
    c.gym.agent.network.width = 10
    c.gym.agent.network.activation = "softplus"

    c.caching.directory = str(cache_dir)
    c.caching.mode = mode
    c.caching.epoch_freq = 1
    c.train.epochs = epochs
    c.train.tf_verbose = 0
    c.output_level = "quiet"
    return c

tf.keras.utils.set_random_seed(42)
c_res = make_cfg(2, r"{str(cache_dir)}", mode="on")
w_res = rw_world.SimpleWorld_Spot_ATM(c_res.world)
vw_res = w_res.clone(samples=50, seed=43)
g_res = rw_gym.VanillaDeepHedgingGym(c_res.gym)
rw_trainer.train(gym=g_res, world=w_res, val_world=vw_res, config=c_res)

res_weights = [w.copy() for w in g_res.get_weights()]
with open(r"{str(res_weights_file)}", "wb") as f:
    pickle.dump(res_weights, f)
print("TF Subprocess resume finished.")
"""
    proc = subprocess.run([sys.executable, "-c", subprocess_code], capture_output=True, text=True)
    if proc.returncode != 0:
        print("TF Subprocess failed:")
        print(proc.stderr)
        raise RuntimeError("Subprocess resume failed!")
        
    with open(res_weights_file, "rb") as f:
        res_weights = pickle.load(f)
        
    weight_diffs = [np.max(np.abs(w1 - w2)) for w1, w2 in zip(cont_weights, res_weights)]
    max_weight_diff = max(weight_diffs)
    print(f"   Max Weight Abs Diff vs Continuous: {max_weight_diff:.4e} (Threshold <= 1e-5)")
    
    # Check Adam optimizer restoration status in cache
    with open(cached_files[0], "rb") as f:
        cache_data = pickle.load(f)
    opt_w = cache_data["gym"].get("opt_weights", [])
    print(f"   Cache contents: keys={list(cache_data.keys())}, progress_data.epoch={cache_data['progress_data'].epoch}, opt_vars={len(opt_w)}, iterations={int(np.asarray(opt_w[0])) if opt_w else None}")
    
    # Cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if res_weights_file.exists():
        res_weights_file.unlink()
        
    print("=" * 60)
    if max_weight_diff <= 1e-5:
        print("VERDICT: TENSORFLOW RESUME TEST PASSED")
        return True
    else:
        print(f"VERDICT: TENSORFLOW RESUME TEST FAILED (Max diff {max_weight_diff:.4e} > 1e-5 due to optimizer state loss across processes)")
        return False

if __name__ == "__main__":
    success = run_tf_warmup_and_resume()
    sys.exit(0 if success else 1)
