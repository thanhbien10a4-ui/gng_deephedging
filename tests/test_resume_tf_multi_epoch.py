#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Multi-Epoch and Multi-Stage Resume Verification Test for TensorFlow Engine
Tests 2 representative configs (Feedforward & Recurrent) with multiple batches/epoch (samples=100, batch_size=32 -> 4 batches/epoch):
1. Continuous 5 epochs
2. 2-stage resume: 2 epochs -> save -> fresh process resume to epoch 5
3. 3-stage resume: 2 epochs -> save -> fresh process resume to epoch 3 -> save -> fresh process resume to epoch 5
Verifies:
- last_weights, Adam m, v, iterations, train_loss, val_loss, and epoch count
- Sample index batch order across all epochs
- State restoration latency and iterator skip latency
- Rejection of incompatible checkpoints (RuntimeError) without silent fallback
"""

import os
import sys
import glob
import pickle
import shutil
import subprocess
import json
import time
from pathlib import Path
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOL = 1e-5

def build_gym_world(cfg_name, epochs, batch_size, cache_dir, mode="on"):
    import tensorflow as tf
    from cdxbasics.config import Config
    import dh_tf.world as W
    import dh_tf.gym as G
    
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
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
    if cfg_name == "rec":
        c.gym.agent.recurrence.states.classic = 1

    c.trainer.train.optimizer.name = "adam"
    c.trainer.train.optimizer.learning_rate = 0.005
    c.trainer.train.batch_size = batch_size
    c.trainer.train.epochs = epochs
    c.trainer.train.tf_verbose = 0
    c.trainer.caching.directory = str(cache_dir)
    c.trainer.caching.mode = mode
    c.trainer.caching.epoch_freq = 1
    c.trainer.output_level = "quiet"

    world = W.SimpleWorld_Spot_ATM(c.world)
    val_world = world.clone(samples=50, seed=43)
    gym = G.VanillaDeepHedgingGym(c.gym)
    return c, world, val_world, gym

def run_phase(cfg_name, epochs, batch_size, cache_dir, out_meta_path=None):
    from dh_tf.trainer import train
    abs_cache = str(Path(cache_dir).resolve())
    c, world, val_world, gym = build_gym_world(cfg_name, epochs, batch_size, abs_cache, mode="on")
    train(gym=gym, world=world, val_world=val_world, config=c.trainer)
    if out_meta_path:
        meta = {
            "last_restore_time": getattr(gym, "last_restore_time", 0.0),
            "last_skip_time": getattr(gym, "last_skip_time", 0.0)
        }
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

def get_cache_file(d):
    files = glob.glob(os.path.join(d, "*.pck"))
    assert len(files) == 1, f"Expected 1 cache file in {d}, found {files}"
    with open(files[0], "rb") as h:
        return pickle.load(h)

def max_diff(list_a, list_b):
    diffs = []
    for a, b in zip(list_a, list_b):
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        if a_arr.size > 0:
            diffs.append(float(np.max(np.abs(a_arr - b_arr))))
    return max(diffs) if diffs else 0.0

def verify_batch_indices(epochs=5):
    import tensorflow as tf
    from keras.src.engine import data_adapter
    
    def extract_indices(skip_epochs=0):
        tf.keras.utils.set_random_seed(42)
        idx = tf.range(100, dtype=tf.int32)
        dh = data_adapter.DataHandler(x=idx, batch_size=32, initial_epoch=0, epochs=epochs, shuffle=True)
        if skip_epochs > 0:
            dh._initial_epoch = skip_epochs
            dh._dataset = dh._dataset.skip(skip_epochs * dh.inferred_steps)
        res = {}
        for ep, it in dh.enumerate_epochs():
            batches = [next(it).numpy().tolist() for _ in range(dh.inferred_steps)]
            res[ep] = batches
        return res

    continuous = extract_indices(skip_epochs=0)
    resumed_2 = extract_indices(skip_epochs=2)
    resumed_3 = extract_indices(skip_epochs=3)
    
    for ep in range(2, epochs):
        assert continuous[ep] == resumed_2[ep], f"Batch index mismatch at ep {ep} for 2->5"
    for ep in range(3, epochs):
        assert continuous[ep] == resumed_3[ep], f"Batch index mismatch at ep {ep} for 3->5"
    
    return {
        "continuous_epochs": list(continuous.keys()),
        "verified_identical": True,
        "sample_counts_per_epoch": [len(batches) for batches in continuous.values()]
    }

def run_test_suite(out_dir):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (out_dir / "work").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    
    py = sys.executable
    script_path = str(Path(__file__).resolve())
    
    print("=" * 70)
    print("STARTING TF MULTI-EPOCH RESUME VERIFICATION (5 EPOCHS, 4 BATCHES/EPOCH)")
    print("=" * 70)
    
    batch_idx_info = verify_batch_indices(5)
    print(f"Sample-index batch ordering verified: {batch_idx_info['verified_identical']} across all epochs.")
    
    suite_results = {
        "batch_index_verification": batch_idx_info,
        "configs": {}
    }
    
    all_pass = True
    
    for cfg in ["ff", "rec"]:
        print(f"\n--- Testing Config: {cfg.upper()} (batch_size=32, 4 batches/epoch) ---")
        dir_cont = work_dir / f"{cfg}_cont5"
        dir_res25 = work_dir / f"{cfg}_res25"
        dir_res235 = work_dir / f"{cfg}_res235"
        
        for d in [dir_cont, dir_res25, dir_res235]:
            if d.exists():
                shutil.rmtree(d)
                
        # 1. Continuous 5 epochs
        print("  1. Running Continuous 5 epochs...")
        proc_cont = subprocess.run([py, script_path, "run_phase", cfg, "5", "32", str(dir_cont)], capture_output=True, text=True)
        assert proc_cont.returncode == 0, f"Continuous run failed: {proc_cont.stderr}"
        cache_cont = get_cache_file(str(dir_cont))
        
        # 2. 2-stage resume: 2 -> 5
        print("  2. Running 2-stage resume (2 epochs -> fresh process to 5)...")
        proc_ep2 = subprocess.run([py, script_path, "run_phase", cfg, "2", "32", str(dir_res25)], capture_output=True, text=True)
        assert proc_ep2.returncode == 0, f"Epoch 2 run failed: {proc_ep2.stderr}"
        
        meta25 = work_dir / f"{cfg}_meta25.json"
        proc_res25 = subprocess.run([py, script_path, "run_phase", cfg, "5", "32", str(dir_res25), str(meta25)], capture_output=True, text=True)
        assert proc_res25.returncode == 0, f"Resume 2->5 failed: {proc_res25.stderr}"
        cache_res25 = get_cache_file(str(dir_res25))
        with open(meta25, "r", encoding="utf-8") as f:
            times25 = json.load(f)
            
        # 3. 3-stage resume: 2 -> 3 -> 5
        print("  3. Running 3-stage resume (2 epochs -> fresh process to 3 -> fresh process to 5)...")
        proc_ep2_35 = subprocess.run([py, script_path, "run_phase", cfg, "2", "32", str(dir_res235)], capture_output=True, text=True)
        assert proc_ep2_35.returncode == 0, f"Epoch 2 of 235 failed: {proc_ep2_35.stderr}"
        
        meta23 = work_dir / f"{cfg}_meta23.json"
        proc_ep3 = subprocess.run([py, script_path, "run_phase", cfg, "3", "32", str(dir_res235), str(meta23)], capture_output=True, text=True)
        assert proc_ep3.returncode == 0, f"Epoch 3 of 235 failed: {proc_ep3.stderr}"
        
        meta35 = work_dir / f"{cfg}_meta35.json"
        proc_res35 = subprocess.run([py, script_path, "run_phase", cfg, "5", "32", str(dir_res235), str(meta35)], capture_output=True, text=True)
        assert proc_res35.returncode == 0, f"Epoch 5 of 235 failed: {proc_res35.stderr}"
        cache_res235 = get_cache_file(str(dir_res235))
        with open(meta35, "r", encoding="utf-8") as f:
            times35 = json.load(f)
            
        # Extract metrics
        w_cont = cache_cont["gym"]["gym_weights"]
        opt_cont = cache_cont["gym"]["opt_weights"]
        pd_cont = cache_cont["progress_data"]
        
        w_res25 = cache_res25["gym"]["gym_weights"]
        opt_res25 = cache_res25["gym"]["opt_weights"]
        pd_res25 = cache_res25["progress_data"]
        
        w_res235 = cache_res235["gym"]["gym_weights"]
        opt_res235 = cache_res235["gym"]["opt_weights"]
        pd_res235 = cache_res235["progress_data"]
        
        n_slots = (len(opt_cont) - 1) // 2
        
        # Diffs: Continuous vs 2-stage (2->5)
        diff_w_25 = float(max_diff(w_cont, w_res25))
        diff_m_25 = float(max_diff(opt_cont[1:1+n_slots], opt_res25[1:1+n_slots]))
        diff_v_25 = float(max_diff(opt_cont[1+n_slots:], opt_res25[1+n_slots:]))
        diff_train_loss_25 = float(abs(pd_cont.losses.training[-1] - pd_res25.losses.training[-1]))
        diff_val_loss_25 = float(abs(pd_cont.losses.val[-1] - pd_res25.losses.val[-1]))
        iter_cont = int(np.asarray(opt_cont[0]))
        iter_res25 = int(np.asarray(opt_res25[0]))
        
        # Diffs: Continuous vs 3-stage (2->3->5)
        diff_w_235 = float(max_diff(w_cont, w_res235))
        diff_m_235 = float(max_diff(opt_cont[1:1+n_slots], opt_res235[1:1+n_slots]))
        diff_v_235 = float(max_diff(opt_cont[1+n_slots:], opt_res235[1+n_slots:]))
        diff_train_loss_235 = float(abs(pd_cont.losses.training[-1] - pd_res235.losses.training[-1]))
        diff_val_loss_235 = float(abs(pd_cont.losses.val[-1] - pd_res235.losses.val[-1]))
        iter_res235 = int(np.asarray(opt_res235[0]))
        
        pass_25 = bool(
            diff_w_25 <= TOL and diff_m_25 <= TOL and diff_v_25 <= TOL
            and diff_train_loss_25 <= TOL and diff_val_loss_25 <= TOL
            and iter_cont == iter_res25 == 20
            and pd_cont.epoch + 1 == pd_res25.epoch + 1 == 5
        )
        
        pass_235 = bool(
            diff_w_235 <= TOL and diff_m_235 <= TOL and diff_v_235 <= TOL
            and diff_train_loss_235 <= TOL and diff_val_loss_235 <= TOL
            and iter_cont == iter_res235 == 20
            and pd_cont.epoch + 1 == pd_res235.epoch + 1 == 5
        )
        
        print(f"  Result 2->5: PASS={pass_25} | MaxDiff w={diff_w_25:.2e}, m={diff_m_25:.2e}, v={diff_v_25:.2e}, LossTrain={diff_train_loss_25:.2e}")
        print(f"               Iterations: {iter_res25} (expected 20), Epochs: {pd_res25.epoch + 1} (expected 5)")
        print(f"               Timings: Restore={times25['last_restore_time']*1000:.2f}ms, IteratorSkip={times25['last_skip_time']*1000:.2f}ms")
        print(f"  Result 2->3->5: PASS={pass_235} | MaxDiff w={diff_w_235:.2e}, m={diff_m_235:.2e}, v={diff_v_235:.2e}, LossTrain={diff_train_loss_235:.2e}")
        print(f"                 Iterations: {iter_res235} (expected 20), Epochs: {pd_res235.epoch + 1} (expected 5)")
        print(f"                 Timings: Restore={times35['last_restore_time']*1000:.2f}ms, IteratorSkip={times35['last_skip_time']*1000:.2f}ms")
        
        if not (pass_25 and pass_235):
            all_pass = False
            
        suite_results["configs"][cfg] = {
            "continuous_5_epochs": {
                "epochs": pd_cont.epoch + 1,
                "iterations": iter_cont,
                "train_loss": float(pd_cont.losses.training[-1]),
                "val_loss": float(pd_cont.losses.val[-1])
            },
            "resume_2_to_5": {
                "pass": pass_25,
                "epochs": pd_res25.epoch + 1,
                "iterations": iter_res25,
                "last_weights_maxdiff": diff_w_25,
                "adam_m_maxdiff": diff_m_25,
                "adam_v_maxdiff": diff_v_25,
                "train_loss_diff": diff_train_loss_25,
                "val_loss_diff": diff_val_loss_25,
                "restore_time_sec": times25["last_restore_time"],
                "iterator_skip_time_sec": times25["last_skip_time"]
            },
            "resume_2_to_3_to_5": {
                "pass": pass_235,
                "epochs": pd_res235.epoch + 1,
                "iterations": iter_res235,
                "last_weights_maxdiff": diff_w_235,
                "adam_m_maxdiff": diff_m_235,
                "adam_v_maxdiff": diff_v_235,
                "train_loss_diff": diff_train_loss_235,
                "val_loss_diff": diff_val_loss_235,
                "restore_time_sec": times35["last_restore_time"],
                "iterator_skip_time_sec": times35["last_skip_time"]
            }
        }
        
    # Test Incompatible Checkpoint Exception Handling
    print("\n--- Testing Incompatible Checkpoint Exception Handling ---")
    incompat_dir = work_dir / "incompat_test"
    if incompat_dir.exists():
        shutil.rmtree(incompat_dir)
        
    # Generate valid cache
    _ = subprocess.run([py, script_path, "run_phase", "ff", "1", "32", str(incompat_dir)], capture_output=True, text=True)
    c_files = glob.glob(str(incompat_dir / "*.pck"))
    assert len(c_files) == 1
    with open(c_files[0], "rb") as h:
        raw_cache = pickle.load(h)
    # Strip opt_weights
    raw_cache["gym"]["opt_weights"] = []
    with open(c_files[0], "wb") as h:
        pickle.dump(raw_cache, h)
        
    # Run trainer.train and verify it raises RuntimeError
    proc_incompat = subprocess.run([py, script_path, "run_phase", "ff", "2", "32", str(incompat_dir)], capture_output=True, text=True)
    incompat_rejected = (proc_incompat.returncode != 0) and ("RuntimeError" in proc_incompat.stderr or "Cache restoration error" in proc_incompat.stdout or "Cache restoration error" in proc_incompat.stderr)
    print(f"Incompatible checkpoint raised exception and refused silent restart: {incompat_rejected} (return code: {proc_incompat.returncode})")
    
    suite_results["incompatible_checkpoint_check"] = {
        "refused_and_raised_exception": incompat_rejected,
        "exit_code": proc_incompat.returncode,
        "stderr_sample": proc_incompat.stderr[-500:] if proc_incompat.stderr else ""
    }
    
    suite_results["all_pass"] = all_pass and incompat_rejected
    
    # Save json summary
    json_path = out_dir / "multi_epoch_resume_summary.json"
    def _default_json(o):
        if hasattr(o, 'item'):
            return o.item()
        if hasattr(o, '__float__'):
            return float(o)
        return str(o)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2, default=_default_json)
        
    print(f"\nSaved summary to: {json_path}")
    print("=" * 70)
    print(f"FINAL MULTI-EPOCH RESUME VERDICT: {'ALL PASS' if suite_results['all_pass'] else 'FAIL'}")
    print("=" * 70)
    
    # Clean temporary work files
    shutil.rmtree(work_dir, ignore_errors=True)
    return suite_results["all_pass"]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_phase":
        meta_p = sys.argv[6] if len(sys.argv) > 6 else None
        run_phase(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], meta_p)
        sys.exit(0)
    target_dir = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else str((ROOT / "results" / "windows_tf_verification_20261106_175449").resolve())
    ok = run_test_suite(target_dir)
    sys.exit(0 if ok else 1)
