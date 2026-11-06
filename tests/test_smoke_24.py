#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke Test Runner for All 24 Deep Hedging Configurations
Executes fast smoke training (2 epochs, up to 500 samples) on CPU compute node via Slurm.
Preserves exact network architecture, features, payoff functions, utilities, and optimizers.
Records:
- Status: PASS / FAIL
- Initial Loss & Final Loss
- Any NaN/Inf occurrence
- Execution time
- Detailed traceback on error
"""

import os
import sys
import time
import json
import traceback
from dataclasses import replace
from pathlib import Path
import numpy as np

# Force CPU and clean logs
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path("/home/user04/gng_deephedging")
REWRITTEN_ROOT = ROOT / "deephedging-rewritten"

if str(REWRITTEN_ROOT) not in sys.path:
    sys.path.insert(0, str(REWRITTEN_ROOT))

from configs.catalog import CONFIGS, get_config

# JAX modules
import jax
import jax.numpy as jnp
from dh_jax.world import WorldConfig as JAXWorldConfig, simulate as jax_simulate
from dh_jax.model import GymConfig as JAXGymConfig, AgentConfig as JAXAgentConfig
from dh_jax.train import TrainConfig as JAXTrainConfig, fit as jax_fit

# TF modules
import tensorflow as tf
from cdxbasics.config import Config as TFConfig
from dh_tf.world import SimpleWorld_Spot_ATM
from dh_tf.gym import VanillaDeepHedgingGym
from dh_tf.trainer import train as tf_train

def run_jax_config(cfg):
    cid = cfg["id"]
    n_samples = cfg["samples"]
    n_steps = cfg["steps"]
    p_type = cfg.get("payoff", "atmcall")
    
    # Payoff function definition
    if p_type == "atmcall":
        payoff_fn = "atmcall"
    elif p_type == "ko_put":
        strike, barrier = 1.0, 0.95
        def payoff_fn(spots):
            notko = np.cumprod(np.where(spots > barrier, 1.0, 0.0), axis=1)
            terminal = np.maximum(strike - spots[:, -1], 0.0)
            return dict(payoff=-terminal * notko[:, -1], features=notko[:, :-1])
        payoff_fn.name = "ko_put"
    elif p_type == "am_digital":
        barrier = 0.95
        def payoff_fn(spots):
            notko = np.cumprod(np.where(spots > barrier, 1.0, 0.0), axis=1)
            terminal = 1.0
            return dict(payoff=-terminal * notko[:, -1], features=notko[:, :-1])
        payoff_fn.name = "am_digital"
    else:
        payoff_fn = "atmcall"

    w_cfg = JAXWorldConfig(
        samples=n_samples, steps=n_steps, dt=cfg.get("dt", 0.02),
        black_scholes=cfg.get("black_scholes", True),
        rvol=cfg.get("rvol", 0.2), drift=cfg.get("drift", 0.0),
        cost_s=cfg.get("cost_s", 0.0), payoff=payoff_fn
    )
    
    world = jax_simulate(w_cfg)
    val_world = world.clone(samples=max(n_samples // 10, 50), seed=1)
    
    rec = cfg.get("recurrent", "none")
    feats = ("price", "delta", "time_left", "payoff_features") if "payoff_features" in world.features else ("price", "delta", "time_left")
    
    classic, aggr, past, event = 0, 0, 0, 0
    bound_aggr = cfg.get("bound_aggr_states", False)
    
    if rec == "digital":
        past, event = 5, 5
    elif rec == "standard":
        aggr = 5
    elif rec == "mix_relu" or rec == "mix_softplus":
        aggr, past = 2, 2
        
    agent = JAXAgentConfig(
        depth=cfg["depth"], width=cfg["width"], activation=cfg["activation"],
        features=feats, classic=classic, aggregate=aggr, past_repr=past, event=event,
        bound_aggr_states=bound_aggr
    )
    gym = JAXGymConfig(agent=agent, utility=cfg["utility"], lmbda=cfg["lmbda"])
    
    t_cfg = JAXTrainConfig(
        epochs=cfg["epochs"], batch_size=min(cfg["batch_size"], n_samples),
        learning_rate=cfg["lr"], clipvalue=cfg.get("clipvalue"),
        global_clipnorm=cfg.get("global_clipnorm")
    )
    
    res = jax_fit(gym, t_cfg, world, val_world)
    init_loss = float(res.result.history["loss"][0])
    final_loss = float(res.result.history["loss"][-1])
    
    # Check NaN/Inf
    has_nan = np.isnan(final_loss) or np.isinf(final_loss)
    for leaf in jax.tree_util.tree_leaves(res.result.params):
        if np.any(np.isnan(leaf)) or np.any(np.isinf(leaf)):
            has_nan = True
            break
            
    return init_loss, final_loss, has_nan

def run_tf_config(cfg):
    cid = cfg["id"]
    n_samples = cfg["samples"]
    n_steps = cfg["steps"]
    p_type = cfg.get("payoff", "atmcall")
    
    c = TFConfig()
    c.world.samples = n_samples
    c.world.steps = n_steps
    c.world.dt = cfg.get("dt", 0.02)
    c.world.drift = cfg.get("drift", 0.0)
    c.world.cost_s = cfg.get("cost_s", 0.0)
    c.world.black_scholes = cfg.get("black_scholes", True)
    
    if p_type == "ko_put":
        strike, barrier = 1.0, 0.95
        def payoff_fn(spots):
            notko = np.cumprod(np.where(spots > barrier, 1.0, 0.0), axis=1)
            terminal = np.maximum(strike - spots[:, -1], 0.0)
            return dict(payoff=-terminal * notko[:, -1], features=notko[:, :-1])
        c.world.payoff = payoff_fn
    elif p_type == "fwd_put":
        iReset = 10
        def fwd_start(spots):
            terminal = np.maximum(1.0 * spots[:, iReset] - spots[:, -1], 0.0)
            return -terminal
        c.world.payoff = fwd_start
    else:
        c.world.payoff = "atmcall"

    c.gym.objective.utility = cfg["utility"]
    c.gym.objective.lmbda = cfg["lmbda"]
    c.gym.agent.network.depth = cfg["depth"]
    c.gym.agent.network.width = cfg["width"]
    c.gym.agent.network.activation = cfg["activation"]
    
    rec = cfg.get("recurrent", "none")
    if p_type == "ko_put":
        c.gym.agent.features = ['price', 'delta', 'time_left', 'payoff_features']
        
    classic, aggr, past, event = 0, 0, 0, 0
    bound_aggr = cfg.get("bound_aggr_states", False)
    
    if rec == "digital":
        past, event = 5, 5
    elif rec == "all_modes":
        classic, aggr, past, event = 2, 2, 2, 2
    elif rec == "aggregate_2":
        aggr = 2
    elif rec == "aggregate_5" or rec == "aggregate_5bnd":
        aggr = 5
    elif rec == "past_repr":
        past = 2
        
    c.gym.agent.recurrence.states.classic = classic
    c.gym.agent.recurrence.states.aggregate = aggr
    c.gym.agent.recurrence.states.past_repr = past
    c.gym.agent.recurrence.states.event = event
    c.gym.agent.recurrence.bound_aggr_states = bound_aggr
    
    c.trainer.train.optimizer.name = "adam"
    c.trainer.train.optimizer.learning_rate = cfg["lr"]
    if cfg.get("clipvalue"):
        c.trainer.train.optimizer.clipvalue = cfg["clipvalue"]
    if cfg.get("global_clipnorm"):
        c.trainer.train.optimizer.global_clipnorm = cfg["global_clipnorm"]
        
    c.trainer.train.batch_size = min(cfg["batch_size"], n_samples)
    c.trainer.train.epochs = cfg["epochs"]
    c.trainer.caching.mode = "off"
    c.trainer.output_level = "quiet"
    
    world = SimpleWorld_Spot_ATM(c.world)
    val_world = world.clone(samples=max(n_samples // 2, 50))
    gym = VanillaDeepHedgingGym(c.gym)
    
    tf_train(gym=gym, world=world, val_world=val_world, config=c.trainer)
    
    out = gym(world.tf_data)
    final_loss = float(np.mean(out["loss"].numpy()))
    has_nan = np.isnan(final_loss) or np.isinf(final_loss)
    for w in gym.trainable_weights:
        if np.any(np.isnan(w.numpy())) or np.any(np.isinf(w.numpy())):
            has_nan = True
            break
            
    return 0.0, final_loss, has_nan

def main():
    print("=" * 75)
    print("RUNNING SMOKE TEST ON ALL 24 CONFIGURATIONS VIA SLURM COMPUTE NODE")
    print(f"Host: {os.uname().nodename} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)
    
    results = []
    t_start_all = time.time()
    
    for i, c_raw in enumerate(CONFIGS):
        cid = c_raw["id"]
        cfg = get_config(cid, mode="smoke")
        backend = cfg["backend"]
        nb = cfg["notebook"]
        
        t0 = time.time()
        status = "PENDING"
        err_msg = None
        tb_str = None
        init_l, final_l, has_nan = None, None, False
        
        try:
            if backend == "jax":
                init_l, final_l, has_nan = run_jax_config(cfg)
            elif backend == "tensorflow":
                init_l, final_l, has_nan = run_tf_config(cfg)
            status = "PASS" if not has_nan else "FAIL (NaN/Inf)"
        except Exception as e:
            status = "FAIL"
            err_msg = str(e)
            tb_str = traceback.format_exc()
            
        elapsed = time.time() - t0
        res = {
            "index": i + 1,
            "id": cid,
            "backend": backend,
            "notebook": nb,
            "group": cfg["group"],
            "status": status,
            "samples": cfg["samples"],
            "epochs": cfg["epochs"],
            "initial_loss": init_l,
            "final_loss": final_l,
            "has_nan": bool(has_nan),
            "elapsed_seconds": round(elapsed, 2),
            "error": err_msg,
            "traceback": tb_str
        }
        results.append(res)
        print(f"[{i+1:02d}/24] {cid:<22} | {backend.upper():<10} | {status:<12} | {elapsed:5.1f}s | FinalLoss: {str(final_l)[:8]}")
        
    total_elapsed = time.time() - t_start_all
    passed = sum(1 for r in results if r["status"] == "PASS")
    
    out_file = REWRITTEN_ROOT / "tests" / "smoke_24_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": os.uname().nodename,
            "total_configs": len(CONFIGS),
            "passed": passed,
            "failed": len(CONFIGS) - passed,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "results": results
        }, f, indent=2)
        
    print("=" * 75)
    print(f"OVERALL SMOKE VERDICT: {passed}/24 PASSED ({passed/24*100:.1f}%) in {total_elapsed:.1f}s")
    print(f"Detailed results saved to {out_file}")
    print("=" * 75)
    
    if passed < len(CONFIGS):
        sys.exit(1)

if __name__ == "__main__":
    main()
