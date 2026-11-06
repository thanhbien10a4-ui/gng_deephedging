#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical 24 Deep Hedging Configurations Catalog
Defines exact hyperparameter mappings for all 7 notebooks and 24 configurations.
Supports both 'original' (full benchmark hyperparameters) and 'smoke' mode (fast pipeline test).
"""

CONFIGS = [
    # 1. blackscholes (5 variants, JAX)
    {
        "id": "bs_base", "group": "blackscholes", "notebook": "blackscholes.ipynb", "cell": "Cell 4", "backend": "jax",
        "samples": 50000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.1, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "cvar", "lmbda": 1.0,
        "depth": 5, "width": 40, "activation": "softplus",
        "recurrent": "none", "batch_size": 256, "lr": 0.005, "epochs": 400, "test_samples": 5000,
        "clipvalue": None, "global_clipnorm": None
    },
    {
        "id": "bs_exp2", "group": "blackscholes", "notebook": "blackscholes.ipynb", "cell": "Cell 9", "backend": "jax",
        "samples": 50000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.0, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "exp2", "lmbda": 10.0,
        "depth": 5, "width": 40, "activation": "softplus",
        "recurrent": "none", "batch_size": 256, "lr": 0.005, "epochs": 400, "test_samples": 5000,
        "clipvalue": None, "global_clipnorm": None
    },
    {
        "id": "bs_nodrift", "group": "blackscholes", "notebook": "blackscholes.ipynb", "cell": "Cell 14", "backend": "jax",
        "samples": 50000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.0, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "cvar", "lmbda": 1.0,
        "depth": 5, "width": 40, "activation": "softplus",
        "recurrent": "none", "batch_size": 256, "lr": 0.005, "epochs": 600, "test_samples": 5000,
        "clipvalue": 0.5, "global_clipnorm": 1.0
    },
    {
        "id": "bs_nodrift_exp2", "group": "blackscholes", "notebook": "blackscholes.ipynb", "cell": "Cell 19", "backend": "jax",
        "samples": 50000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.0, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "exp2", "lmbda": 10.0,
        "depth": 5, "width": 40, "activation": "softplus",
        "recurrent": "none", "batch_size": 256, "lr": 0.001, "epochs": 600, "test_samples": 5000,
        "clipvalue": 0.5, "global_clipnorm": 1.0
    },
    {
        "id": "bs_small", "group": "blackscholes", "notebook": "blackscholes.ipynb", "cell": "Cell 24", "backend": "jax",
        "samples": 10000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.1, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "cvar", "lmbda": 2.0,
        "depth": 5, "width": 40, "activation": "softplus",
        "recurrent": "none", "batch_size": 256, "lr": 0.005, "epochs": 10, "test_samples": 1000,
        "clipvalue": None, "global_clipnorm": None
    },

    # 2. trainer-bs (2 variants, TensorFlow)
    {
        "id": "trainer_bs_coarse", "group": "trainer-bs", "notebook": "trainer-bs.ipynb", "cell": "Cell 2, 4", "backend": "tensorflow",
        "samples": 10000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.0, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "cvar", "lmbda": 1.0,
        "depth": 3, "width": 20, "activation": "softplus",
        "recurrent": "none", "batch_size": 2000, "lr": 0.005, "epochs": 20000, "test_samples": 5000,
        "clipvalue": None, "global_clipnorm": None
    },
    {
        "id": "trainer_bs_fine", "group": "trainer-bs", "notebook": "trainer-bs.ipynb", "cell": "Cell 2, 7", "backend": "tensorflow",
        "samples": 10000, "steps": 100, "dt": 0.004, "rvol": 0.2, "drift": 0.0, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "cvar", "lmbda": 1.0,
        "depth": 3, "width": 20, "activation": "softplus",
        "recurrent": "none", "batch_size": 2000, "lr": 0.005, "epochs": 20000, "test_samples": 5000,
        "clipvalue": None, "global_clipnorm": None
    },

    # 3. trainer (2 variants, JAX)
    {
        "id": "trainer_bs", "group": "trainer", "notebook": "trainer.ipynb", "cell": "Cell 3", "backend": "jax",
        "samples": 10000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.0, "cost_s": 0.0, "black_scholes": True,
        "payoff": "atmcall", "utility": "cvar", "lmbda": 1.0,
        "depth": 3, "width": 20, "activation": "softplus",
        "recurrent": "none", "batch_size": 32, "lr": 0.005, "epochs": 800, "test_samples": 1000,
        "clipvalue": None, "global_clipnorm": None
    },
    {
        "id": "trainer_sv", "group": "trainer", "notebook": "trainer.ipynb", "cell": "Cell 7", "backend": "jax",
        "samples": 10000, "steps": 20, "dt": 0.02, "rvol": 0.2, "drift": 0.0, "cost_s": 0.0, "black_scholes": False,
        "payoff": "atmcall", "utility": "cvar", "lmbda": 1.0,
        "depth": 3, "width": 20, "activation": "softplus",
        "recurrent": "none", "batch_size": 32, "lr": 0.001, "epochs": 800, "test_samples": 1000,
        "clipvalue": 1.0, "global_clipnorm": 1.0
    },

    # 4. trainer-american_digital (5 variants, JAX)
    {
        "id": "am_dig_none", "group": "american_digital", "notebook": "trainer-american_digital.ipynb", "cell": "Cell 2, 4", "backend": "jax",
        "samples": 20000, "steps": 100, "dt": 0.02, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "am_digital", "utility": "exp2", "lmbda": 1.0, "recurrent": "none",
        "depth": 5, "width": 50, "activation": "softplus",
        "batch_size": 32, "lr": 0.0001, "epochs": 500, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "am_dig_digital", "group": "american_digital", "notebook": "trainer-american_digital.ipynb", "cell": "Cell 2, 4", "backend": "jax",
        "samples": 20000, "steps": 100, "dt": 0.02, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "am_digital", "utility": "exp2", "lmbda": 1.0, "recurrent": "digital",
        "depth": 5, "width": 50, "activation": "softplus",
        "batch_size": 32, "lr": 0.0001, "epochs": 500, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "am_dig_standard", "group": "american_digital", "notebook": "trainer-american_digital.ipynb", "cell": "Cell 2, 4", "backend": "jax",
        "samples": 20000, "steps": 100, "dt": 0.02, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "am_digital", "utility": "exp2", "lmbda": 1.0, "recurrent": "standard",
        "depth": 5, "width": 50, "activation": "softplus",
        "batch_size": 32, "lr": 0.0001, "epochs": 500, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "am_dig_mix_relu", "group": "american_digital", "notebook": "trainer-american_digital.ipynb", "cell": "Cell 2, 4", "backend": "jax",
        "samples": 20000, "steps": 100, "dt": 0.02, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "am_digital", "utility": "exp2", "lmbda": 1.0, "recurrent": "mix_relu",
        "depth": 5, "width": 100, "activation": "relu",
        "batch_size": 32, "lr": 0.0001, "epochs": 500, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "am_dig_mix_softplus", "group": "american_digital", "notebook": "trainer-american_digital.ipynb", "cell": "Cell 2, 4", "backend": "jax",
        "samples": 20000, "steps": 100, "dt": 0.02, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "am_digital", "utility": "exp2", "lmbda": 1.0, "recurrent": "mix_softplus",
        "depth": 5, "width": 100, "activation": "softplus",
        "batch_size": 32, "lr": 0.0001, "epochs": 500, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },

    # 5. trainer-recurrent-barrier (2 variants, TensorFlow)
    {
        "id": "barrier_none", "group": "barrier", "notebook": "trainer-recurrent-barrier.ipynb", "cell": "Cell 2, 4", "backend": "tensorflow",
        "samples": 20000, "steps": 100, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "ko_put", "utility": "exp", "lmbda": 1.01, "recurrent": "none",
        "depth": 5, "width": 50, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 250, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "barrier_digital", "group": "barrier", "notebook": "trainer-recurrent-barrier.ipynb", "cell": "Cell 2, 4", "backend": "tensorflow",
        "samples": 20000, "steps": 100, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "ko_put", "utility": "exp", "lmbda": 1.01, "recurrent": "digital",
        "depth": 5, "width": 50, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 250, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },

    # 6. trainer-recurrent-barrier_smalldt (2 variants, JAX)
    {
        "id": "barrier_sdt_none", "group": "barrier_smalldt", "notebook": "trainer-recurrent-barrier_smalldt.ipynb", "cell": "Cell 2, 4", "backend": "jax",
        "samples": 20000, "steps": 20, "dt": 0.02, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "ko_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "none",
        "depth": 5, "width": 50, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 250, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "barrier_sdt_digital", "group": "barrier_smalldt", "notebook": "trainer-recurrent-barrier_smalldt.ipynb", "cell": "Cell 2, 4", "backend": "jax",
        "samples": 20000, "steps": 20, "dt": 0.02, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "ko_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "digital",
        "depth": 5, "width": 50, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 250, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },

    # 7. trainer-recurrent-fwdstart (6 variants, TensorFlow)
    {
        "id": "fwd_none", "group": "fwdstart", "notebook": "trainer-recurrent-fwdstart.ipynb", "cell": "Cell 2, 3", "backend": "tensorflow",
        "samples": 20000, "steps": 20, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "fwd_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "none",
        "depth": 5, "width": 20, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 2000, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "fwd_all_modes", "group": "fwdstart", "notebook": "trainer-recurrent-fwdstart.ipynb", "cell": "Cell 2, 4", "backend": "tensorflow",
        "samples": 20000, "steps": 20, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "fwd_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "all_modes",
        "depth": 5, "width": 20, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 2000, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "fwd_aggr_2", "group": "fwdstart", "notebook": "trainer-recurrent-fwdstart.ipynb", "cell": "Cell 2, 5", "backend": "tensorflow",
        "samples": 20000, "steps": 20, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "fwd_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "aggregate_2",
        "depth": 5, "width": 20, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 2000, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "fwd_aggr_5bnd", "group": "fwdstart", "notebook": "trainer-recurrent-fwdstart.ipynb", "cell": "Cell 2, 6", "backend": "tensorflow",
        "samples": 20000, "steps": 20, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "fwd_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "aggregate_5bnd", "bound_aggr_states": True,
        "depth": 5, "width": 20, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 2000, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "fwd_aggr_5", "group": "fwdstart", "notebook": "trainer-recurrent-fwdstart.ipynb", "cell": "Cell 2, 7", "backend": "tensorflow",
        "samples": 20000, "steps": 20, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "fwd_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "aggregate_5", "bound_aggr_states": False,
        "depth": 5, "width": 20, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 2000, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
    {
        "id": "fwd_past_repr", "group": "fwdstart", "notebook": "trainer-recurrent-fwdstart.ipynb", "cell": "Cell 2, 8", "backend": "tensorflow",
        "samples": 20000, "steps": 20, "dt": 0.004, "black_scholes": True, "drift": 0.0, "cost_s": 0.0,
        "payoff": "fwd_put", "utility": "cvar", "lmbda": 1.01, "recurrent": "past_repr",
        "depth": 5, "width": 20, "activation": "softplus",
        "batch_size": 32, "lr": 0.001, "epochs": 2000, "test_samples": 10000,
        "clipvalue": 0.9, "global_clipnorm": 1.0
    },
]

CONFIG_MAP = {c["id"]: c for c in CONFIGS}

def get_config(config_id: str, mode: str = "original") -> dict:
    """
    Retrieve config dictionary by ID in either 'original' or 'smoke' mode.
    - 'original': exact source hyperparameters (epochs, samples, batch size, etc.)
    - 'smoke': reduced epochs (2) and samples (min(samples, 500)) for fast verification.
      All features, architectures, payoffs, utilities, and optimizers are preserved.
    """
    if config_id not in CONFIG_MAP:
        raise KeyError(f"Unknown config_id: {config_id}. Available: {list(CONFIG_MAP.keys())}")
    cfg = dict(CONFIG_MAP[config_id])
    if mode == "smoke":
        cfg["samples"] = min(cfg["samples"], 500)
        cfg["test_samples"] = min(cfg["test_samples"], 200)
        cfg["epochs"] = 2
        # Ensure batch_size does not exceed sample size
        if cfg["batch_size"] and cfg["batch_size"] > cfg["samples"]:
            cfg["batch_size"] = cfg["samples"]
    elif mode != "original":
        raise ValueError(f"Invalid mode: {mode}. Must be 'original' or 'smoke'.")
    return cfg
