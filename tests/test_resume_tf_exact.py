#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF exact-resume test (fresh process).
For a config (ff | rec) and batch size:
  A: continuous 2 epochs                       (process 1, own cache dir)
  B: 1 epoch, checkpoint written               (process 2)
  C: fresh process, load checkpoint, epoch 2   (process 3, same dir as B)
Compares last weights, Adam iterations/m/v, train & val loss of epoch 2 (A vs C).
The finalize() cache contains the *last* weights + optimizer (written before best-weights reset).
Thresholds: weights/slots <= 1e-5 abs, losses <= 1e-5 abs, iterations exact.
Also: legacy checkpoint without optimizer state must be refused.
"""
import os, sys, glob, pickle, shutil, subprocess, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

TOL = 1e-5

def build(cfg_name, epochs, bs, cache_dir):
    import tensorflow as tf
    from cdxbasics.config import Config
    import dh_tf.world as W, dh_tf.gym as G
    c = Config()
    c.world.samples = 100; c.world.steps = 5; c.world.dt = 0.02; c.world.drift = 0.0
    c.world.cost_s = 0.0; c.world.black_scholes = True; c.world.seed = 42
    c.gym.objective.utility = "cvar"; c.gym.objective.lmbda = 1.0
    c.gym.agent.network.depth = 2; c.gym.agent.network.width = 10; c.gym.agent.network.activation = "softplus"
    if cfg_name == "rec":
        c.gym.agent.recurrence.states.classic = 1
    c.trainer.train.optimizer.name = "adam"
    c.trainer.train.optimizer.learning_rate = 0.005
    c.trainer.train.batch_size = bs
    c.trainer.train.epochs = epochs
    c.trainer.caching.directory = cache_dir
    c.trainer.caching.mode = "on"
    c.trainer.caching.epoch_freq = 1
    c.trainer.output_level = "text"
    tf.keras.utils.set_random_seed(42)
    world = W.SimpleWorld_Spot_ATM(c.world)
    val = world.clone(samples=50, seed=43)
    gym = G.VanillaDeepHedgingGym(c.gym)
    return c, world, val, gym

def phase(cfg_name, epochs, bs, cache_dir):
    from dh_tf.trainer import train
    c, world, val, gym = build(cfg_name, epochs, bs, cache_dir)
    train(gym=gym, world=world, val_world=val, config=c.trainer)
    print("IS_RECURRENT", gym.agent.is_recurrent)

def load_cache(d):
    f = glob.glob(os.path.join(d, "*.pck"))
    assert len(f) == 1, f
    with open(f[0], "rb") as h:
        return pickle.load(h)

def run(args, log):
    p = subprocess.run([sys.executable, os.path.abspath(__file__)] + args, capture_output=True, text=True, cwd=ROOT)
    log.write(f"\n### {' '.join(args)} (exit {p.returncode})\n{p.stdout}\n--- stderr ---\n{p.stderr[-3000:]}\n")
    if p.returncode != 0:
        raise RuntimeError(f"phase failed: {args}")
    return p.stdout

def maxdiff(a, b):
    return max(float(np.max(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)))) if np.size(x) else 0.0 for x, y in zip(a, b))

def compare(cfg_name, bs, work, log):
    dA, dB = os.path.join(work, f"{cfg_name}_{bs}_A"), os.path.join(work, f"{cfg_name}_{bs}_B")
    for d in (dA, dB):
        shutil.rmtree(d, ignore_errors=True)
    outA = run(["phase", cfg_name, "2", str(bs), dA], log)
    run(["phase", cfg_name, "1", str(bs), dB], log)
    after1 = load_cache(dB)
    outC = run(["phase", cfg_name, "2", str(bs), dB], log)
    A, C = load_cache(dA), load_cache(dB)
    ga, gc = A["gym"], C["gym"]
    oa, oc = ga["opt_weights"], gc["opt_weights"]
    n = (len(oa) - 1) // 2
    pa, pc = A["progress_data"], C["progress_data"]
    r = dict(
        config=cfg_name, batch_size=bs, recurrent=("IS_RECURRENT True" in outA),
        resumed_from_cache=("Cache successfully loaded" in outC),
        epoch1_opt_iterations=int(np.asarray(after1["gym"]["opt_weights"][0])),
        epochs_recorded_A=pa.epoch + 1, epochs_recorded_C=pc.epoch + 1,
        iterations_A=int(np.asarray(oa[0])), iterations_C=int(np.asarray(oc[0])),
        last_weights_maxdiff=maxdiff(ga["gym_weights"], gc["gym_weights"]),
        adam_m_maxdiff=maxdiff(oa[1:1 + n], oc[1:1 + n]),
        adam_v_maxdiff=maxdiff(oa[1 + n:], oc[1 + n:]),
        train_loss_A=pa.losses.training[-1], train_loss_C=pc.losses.training[-1],
        val_loss_A=pa.losses.val[-1], val_loss_C=pc.losses.val[-1],
        best_epoch_A=pa.best_epoch, best_epoch_C=pc.best_epoch,
    )
    r["train_loss_absdiff"] = abs(r["train_loss_A"] - r["train_loss_C"])
    r["val_loss_absdiff"] = abs(r["val_loss_A"] - r["val_loss_C"])
    r["PASS"] = bool(r["resumed_from_cache"] and r["epochs_recorded_C"] == 2 and r["iterations_A"] == r["iterations_C"]
                     and r["epoch1_opt_iterations"] > 0
                     and max(r["last_weights_maxdiff"], r["adam_m_maxdiff"], r["adam_v_maxdiff"]) <= TOL
                     and max(r["train_loss_absdiff"], r["val_loss_absdiff"]) <= TOL)
    return r

def legacy_phase(cache_dir):
    """Strip optimizer state from a checkpoint and verify restore_from_cache refuses it."""
    import tensorflow as tf
    c, world, val, gym = build("ff", 2, 100, cache_dir)
    from dh_tf.base import create_optimizer
    cache = load_cache(cache_dir)
    gym(world.tf_data)
    gym.compile(optimizer=create_optimizer(c.trainer.train), loss=dict(loss=lambda y, p: p))
    w0 = [w.copy() for w in gym.get_weights()]
    g = dict(cache["gym"]); g["opt_weights"] = []
    ok_missing = gym.restore_from_cache(g)
    unchanged = maxdiff(w0, gym.get_weights()) == 0.0
    ok_full = gym.restore_from_cache(cache["gym"])
    it = int(gym.optimizer.iterations.numpy())
    print("LEGACY", json.dumps(dict(refused_missing_opt_state=(ok_missing is False), weights_untouched=unchanged,
                                    full_restore=ok_full, iterations_after_full_restore=it)))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "phase":
        phase(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "legacy":
        legacy_phase(sys.argv[2]); sys.exit(0)
    out = os.path.abspath(sys.argv[1])
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "work"); os.makedirs(work, exist_ok=True)
    results = []
    with open(os.path.join(out, "logs", "tf_resume_phases.log"), "w", encoding="utf-8") as log:
        for cfg in ("ff", "rec"):
            for bs in (32, 100):
                results.append(compare(cfg, bs, work, log))
        lo = run(["legacy", os.path.join(work, "ff_32_A")], log)
    legacy = json.loads([l for l in lo.splitlines() if l.startswith("LEGACY")][0][7:])
    all_pass = all(c["PASS"] for c in results) and legacy["refused_missing_opt_state"] and legacy["weights_untouched"] and legacy["full_restore"]
    res = dict(all_pass=all_pass, cases=results, legacy=legacy)
    with open(os.path.join(out, "tf_resume_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    shutil.rmtree(work, ignore_errors=True)
    assert all_pass, "One or more TF resume checks failed!"
