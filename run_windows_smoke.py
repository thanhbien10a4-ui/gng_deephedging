#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Hedging Windows Smoke Test Runner
Executes all 7 notebooks sequentially on CPU Windows:
- 4 JAX notebooks (14 configs) using Python 3.13 (dh-rewritten-jax)
- 3 TensorFlow notebooks (10 configs) using .venv-tf Python 3.10 (dh-rewritten-tf)

Saves results to results/windows_smoke_<YYYYMMDD_HHMMSS>/:
- executed_notebooks/
- logs/
- environment/
- isolated_checkpoints/
- summary.csv & summary.json
"""

import os
import sys
import time
import json
import csv
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

# ---------------------------------------------------------
# Global Setup & Constants
# ---------------------------------------------------------
REPO_DIR = Path(r"D:\deephedging-rewritten")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = REPO_DIR / "results" / f"windows_smoke_{TIMESTAMP}"
EXEC_DIR = RESULTS_DIR / "executed_notebooks"
LOGS_DIR = RESULTS_DIR / "logs"
ENV_DIR = RESULTS_DIR / "environment"
ISOLATED_CKPTS_DIR = RESULTS_DIR / "isolated_checkpoints"

JAX_INTERPRETER = r"C:\Users\Hi Windows 11 Home\AppData\Local\Programs\Python\Python313\python.exe"
TF_INTERPRETER = str(REPO_DIR / r".venv-tf\Scripts\python.exe")

# Environment variables for smoke test
os.environ["DH_MODE"] = "smoke"
os.environ.pop("DH_EPOCHS", None)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["DH_CHECKPOINTS"] = str(ISOLATED_CKPTS_DIR)

NOTEBOOKS = [
    # JAX Group (4 notebooks, 14 configs)
    {
        "name": "trainer.ipynb",
        "backend": "jax",
        "kernel": "dh-rewritten-jax",
        "interpreter": JAX_INTERPRETER,
        "expected_configs": ["trainer_bs", "trainer_sv"],
    },
    {
        "name": "blackscholes.ipynb",
        "backend": "jax",
        "kernel": "dh-rewritten-jax",
        "interpreter": JAX_INTERPRETER,
        "expected_configs": ["bs_base", "bs_exp2", "bs_nodrift", "bs_nodrift_exp2", "bs_small"],
    },
    {
        "name": "trainer-american_digital.ipynb",
        "backend": "jax",
        "kernel": "dh-rewritten-jax",
        "interpreter": JAX_INTERPRETER,
        "expected_configs": ["am_dig_none", "am_dig_digital", "am_dig_standard", "am_dig_mix_relu", "am_dig_mix_softplus"],
    },
    {
        "name": "trainer-recurrent-barrier_smalldt.ipynb",
        "backend": "jax",
        "kernel": "dh-rewritten-jax",
        "interpreter": JAX_INTERPRETER,
        "expected_configs": ["barrier_sdt_none", "barrier_sdt_digital"],
    },
    # TensorFlow Group (3 notebooks, 10 configs)
    {
        "name": "trainer-bs.ipynb",
        "backend": "tensorflow",
        "kernel": "dh-rewritten-tf",
        "interpreter": TF_INTERPRETER,
        "expected_configs": ["trainer_bs_coarse", "trainer_bs_fine"],
    },
    {
        "name": "trainer-recurrent-barrier.ipynb",
        "backend": "tensorflow",
        "kernel": "dh-rewritten-tf",
        "interpreter": TF_INTERPRETER,
        "expected_configs": ["barrier_none", "barrier_digital"],
    },
    {
        "name": "trainer-recurrent-fwdstart.ipynb",
        "backend": "tensorflow",
        "kernel": "dh-rewritten-tf",
        "interpreter": TF_INTERPRETER,
        "expected_configs": ["fwd_none", "fwd_all_modes", "fwd_aggr_2", "fwd_aggr_5bnd", "fwd_aggr_5", "fwd_past_repr"],
    },
]


def capture_environment_info():
    """Records Python versions and pip package lists for both environments."""
    print("Capturing environment details...")
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    
    # JAX Environment
    try:
        py_jax = subprocess.check_output([JAX_INTERPRETER, "-V"], text=True).strip()
        (ENV_DIR / "python_version_jax.txt").write_text(py_jax, encoding="utf-8")
        pip_jax = subprocess.check_output([JAX_INTERPRETER, "-m", "pip", "list"], text=True)
        (ENV_DIR / "packages_jax.txt").write_text(pip_jax, encoding="utf-8")
    except Exception as e:
        print(f"Warning: Failed to capture JAX environment: {e}")

    # TF Environment
    try:
        py_tf = subprocess.check_output([TF_INTERPRETER, "-V"], text=True).strip()
        (ENV_DIR / "python_version_tf.txt").write_text(py_tf, encoding="utf-8")
        pip_tf = subprocess.check_output([TF_INTERPRETER, "-m", "pip", "list"], text=True)
        (ENV_DIR / "packages_tf.txt").write_text(pip_tf, encoding="utf-8")
    except Exception as e:
        print(f"Warning: Failed to capture TF environment: {e}")


def get_query_code(nb_name: str, cell_src: str):
    """Returns query code snippet to extract config metrics if this cell trained a config."""
    s = cell_src
    
    # 1. trainer.ipynb
    if nb_name == "trainer.ipynb":
        if "trainer_bs" in s and "fit(" in s:
            return """
import json, numpy as np
tl = float(bs_fit.result.history["loss"][-1])
vl = float(bs_fit.result.history["val_loss"][-1])
ep = len(bs_fit.result.history["loss"])
res = {
    "id": "trainer_bs",
    "samples": int(world_bs.nSamples),
    "epochs": ep,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl))
}
print("SMOKE_RESULT:" + json.dumps(res))
"""
        elif "trainer_sv" in s and "fit(" in s:
            return """
import json, numpy as np
tl = float(sv_fit.result.history["loss"][-1])
vl = float(sv_fit.result.history["val_loss"][-1])
ep = len(sv_fit.result.history["loss"])
res = {
    "id": "trainer_sv",
    "samples": int(world_cfg_sv.samples),
    "epochs": ep,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl))
}
print("SMOKE_RESULT:" + json.dumps(res))
"""

    # 2. blackscholes.ipynb
    elif nb_name == "blackscholes.ipynb":
        fit_map = [
            ("bs_nodrift_exp2", "nodrift_exp2_fit"),
            ("bs_nodrift", "nodrift_fit"),
            ("bs_base", "base_fit"),
            ("bs_exp2", "exp2_fit"),
            ("bs_small", "small_fit"),
        ]
        for cid, var in fit_map:
            if f'"{cid}"' in s and "fit(" in s:
                return f"""
import json, numpy as np
tl = float({var}.result.history["loss"][-1])
vl = float({var}.result.history["val_loss"][-1])
ep = len({var}.result.history["loss"])
res = {{
    "id": "{cid}",
    "samples": int(world_cfg.samples),
    "epochs": ep,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl))
}}
print("SMOKE_RESULT:" + json.dumps(res))
"""

    # 3. trainer-american_digital.ipynb
    elif nb_name == "trainer-american_digital.ipynb":
        if "configs = {" in s and "fits = {}" in s:
            return """
import json, numpy as np
c_map = {
    "none": "am_dig_none",
    "digital": "am_dig_digital",
    "standard": "am_dig_standard",
    "mix_relu": "am_dig_mix_relu",
    "mix_softplus": "am_dig_mix_softplus"
}
for k, cid in c_map.items():
    f = fits[k]
    tl = float(f.result.history["loss"][-1])
    vl = float(f.result.history["val_loss"][-1])
    ep = len(f.result.history["loss"])
    res = {
        "id": cid,
        "samples": int(world.nSamples),
        "epochs": ep,
        "train_loss": tl,
        "val_loss": vl,
        "is_finite": bool(np.isfinite(tl) and np.isfinite(vl))
    }
    print("SMOKE_RESULT:" + json.dumps(res))
"""

    # 4. trainer-recurrent-barrier_smalldt.ipynb
    elif nb_name == "trainer-recurrent-barrier_smalldt.ipynb":
        if "configs = {" in s and "fits = {}" in s:
            return """
import json, numpy as np
c_map = {
    "none": "barrier_sdt_none",
    "digital": "barrier_sdt_digital"
}
for k, cid in c_map.items():
    f = fits[k]
    tl = float(f.result.history["loss"][-1])
    vl = float(f.result.history["val_loss"][-1])
    ep = len(f.result.history["loss"])
    res = {
        "id": cid,
        "samples": int(world.nSamples),
        "epochs": ep,
        "train_loss": tl,
        "val_loss": vl,
        "is_finite": bool(np.isfinite(tl) and np.isfinite(vl))
    }
    print("SMOKE_RESULT:" + json.dumps(res))
"""

    # 5. trainer-bs.ipynb (TF)
    elif nb_name == "trainer-bs.ipynb":
        if "trainer_bs_coarse" in s and "train(" in s:
            return """
import json, numpy as np
out = gym_coarse(world_coarse.tf_data)
val_out = gym_coarse(val_world_coarse.tf_data)
tl = float(np.mean(out["loss"].numpy()))
vl = float(np.mean(val_out["loss"].numpy()))
w_finite = all(np.all(np.isfinite(w.numpy())) for w in gym_coarse.trainable_weights)
res = {
    "id": "trainer_bs_coarse",
    "samples": int(world_coarse.nSamples),
    "epochs": 2,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl) and w_finite)
}
print("SMOKE_RESULT:" + json.dumps(res))
"""
        elif "trainer_bs_fine" in s and "train(" in s:
            return """
import json, numpy as np
out = gym_fine(world_fine.tf_data)
val_out = gym_fine(val_world_fine.tf_data)
tl = float(np.mean(out["loss"].numpy()))
vl = float(np.mean(val_out["loss"].numpy()))
w_finite = all(np.all(np.isfinite(w.numpy())) for w in gym_fine.trainable_weights)
res = {
    "id": "trainer_bs_fine",
    "samples": int(world_fine.nSamples),
    "epochs": 2,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl) and w_finite)
}
print("SMOKE_RESULT:" + json.dumps(res))
"""

    # 6. trainer-recurrent-barrier.ipynb (TF)
    elif nb_name == "trainer-recurrent-barrier.ipynb":
        if "barrier_none" in s and "train(" in s:
            return """
import json, numpy as np
out = gym_none(world.tf_data)
val_out = gym_none(val_world.tf_data)
tl = float(np.mean(out["loss"].numpy()))
vl = float(np.mean(val_out["loss"].numpy()))
w_finite = all(np.all(np.isfinite(w.numpy())) for w in gym_none.trainable_weights)
res = {
    "id": "barrier_none",
    "samples": int(world.nSamples),
    "epochs": 2,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl) and w_finite)
}
print("SMOKE_RESULT:" + json.dumps(res))
"""
        elif "barrier_digital" in s and "train(" in s:
            return """
import json, numpy as np
out = gym_digital(world.tf_data)
val_out = gym_digital(val_world.tf_data)
tl = float(np.mean(out["loss"].numpy()))
vl = float(np.mean(val_out["loss"].numpy()))
w_finite = all(np.all(np.isfinite(w.numpy())) for w in gym_digital.trainable_weights)
res = {
    "id": "barrier_digital",
    "samples": int(world.nSamples),
    "epochs": 2,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl) and w_finite)
}
print("SMOKE_RESULT:" + json.dumps(res))
"""

    # 7. trainer-recurrent-fwdstart.ipynb (TF)
    elif nb_name == "trainer-recurrent-fwdstart.ipynb":
        fwd_order = [
            ("fwd_aggr_5bnd", "Training fwd_aggr_5bnd"),
            ("fwd_aggr_5", "Training fwd_aggr_5"),
            ("fwd_aggr_2", "Training fwd_aggr_2"),
            ("fwd_all_modes", "Training fwd_all_modes"),
            ("fwd_past_repr", "Training fwd_past_repr"),
            ("fwd_none", "Training fwd_none"),
        ]
        for cid, marker in fwd_order:
            if marker in s and "train(" in s:
                return f"""
import json, numpy as np
out = gym(world.tf_data)
val_out = gym(val_world.tf_data)
tl = float(np.mean(out["loss"].numpy()))
vl = float(np.mean(val_out["loss"].numpy()))
w_finite = all(np.all(np.isfinite(w.numpy())) for w in gym.trainable_weights)
res = {{
    "id": "{cid}",
    "samples": int(world.nSamples),
    "epochs": 2,
    "train_loss": tl,
    "val_loss": vl,
    "is_finite": bool(np.isfinite(tl) and np.isfinite(vl) and w_finite)
}}
print("SMOKE_RESULT:" + json.dumps(res))
"""

    return None


def run_single_notebook(spec: dict):
    """Executes a single notebook using nbclient with isolated kernel and records logs/metrics."""
    nb_name = spec["name"]
    backend = spec["backend"]
    kernel = spec["kernel"]
    interpreter = spec["interpreter"]
    expected = spec["expected_configs"]
    
    src_path = REPO_DIR / nb_name
    exec_path = EXEC_DIR / nb_name
    log_path = LOGS_DIR / f"{nb_name.replace('.ipynb', '')}.log"
    
    print(f"\n{'='*75}")
    print(f"RUNNING NOTEBOOK: {nb_name} [{backend.upper()}]")
    print(f"Kernel: {kernel} | Interpreter: {interpreter}")
    print(f"Expected configs: {', '.join(expected)}")
    print(f"{'='*75}")
    
    log_lines = []
    def log(msg: str):
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t_str}] {msg}"
        log_lines.append(line)
        print(line)

    log(f"Starting execution of {nb_name}")
    log(f"Backend: {backend} | Kernel: {kernel}")
    log(f"DH_MODE={os.environ.get('DH_MODE')} | DH_CHECKPOINTS={os.environ.get('DH_CHECKPOINTS')}")

    nb = nbformat.read(src_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name=kernel,
        resources={"metadata": {"path": str(REPO_DIR)}}
    )

    t_nb_start = time.time()
    config_results = {}
    nb_error = None

    try:
        with client.setup_kernel():
            num_cells = len(client.nb.cells)
            log(f"Kernel initialized successfully. Total cells to execute: {num_cells}")
            
            for idx in range(num_cells):
                cell = client.nb.cells[idx]
                if cell.cell_type != "code":
                    continue
                
                c_snippet = cell.source.strip().splitlines()[0] if cell.source.strip() else "(empty)"
                log(f"Executing Code Cell {idx}: {c_snippet[:70]}...")
                t_cell_start = time.time()
                
                try:
                    client.execute_cell(cell, idx)
                    cell_dur = time.time() - t_cell_start
                    log(f"Cell {idx} completed in {cell_dur:.2f}s")
                    
                    # Log text outputs
                    for out in cell.outputs:
                        if "text" in out:
                            for l in out["text"].strip().splitlines():
                                log(f"  [stdout] {l}")
                        elif "traceback" in out:
                            for l in out["traceback"]:
                                log(f"  [traceback] {l}")
                    
                    # Check if cell trained a config
                    query_code = get_query_code(nb_name, cell.source)
                    if query_code:
                        log(f"  -> Querying kernel for configuration metrics...")
                        t_q_start = time.time()
                        q_cell = nbformat.v4.new_code_cell(query_code)
                        client.nb.cells.append(q_cell)
                        q_idx = len(client.nb.cells) - 1
                        client.execute_cell(q_cell, q_idx)
                        
                        # Parse SMOKE_RESULT lines
                        for out in q_cell.outputs:
                            if "text" in out:
                                for l in out["text"].splitlines():
                                    if "SMOKE_RESULT:" in l:
                                        payload = l.split("SMOKE_RESULT:", 1)[1].strip()
                                        m = json.loads(payload)
                                        cid = m["id"]
                                        m["cell_duration_sec"] = round(cell_dur, 2)
                                        config_results[cid] = m
                                        log(f"  [METRIC] Config '{cid}': samples={m['samples']}, epochs={m['epochs']}, train_loss={m['train_loss']:.6f}, val_loss={m['val_loss']:.6f}, finite={m['is_finite']}")
                            if "traceback" in out:
                                log(f"  [Query Error] {' '.join(out['traceback'])}")
                        
                        # Remove query cell to keep executed notebook pristine
                        client.nb.cells.pop()

                except Exception as cell_exc:
                    log(f"ERROR in Cell {idx}: {cell_exc}")
                    log(traceback.format_exc())
                    nb_error = str(cell_exc)
                    break

    except Exception as nb_exc:
        log(f"FATAL Kernel/Client ERROR: {nb_exc}")
        log(traceback.format_exc())
        nb_error = str(nb_exc)

    t_nb_duration = time.time() - t_nb_start
    log(f"Notebook {nb_name} finished in {t_nb_duration:.2f}s. Captured {len(config_results)}/{len(expected)} configs.")
    
    # Save executed notebook (with all executed outputs, errors, charts)
    try:
        nbformat.write(client.nb, exec_path)
        log(f"Saved executed notebook to: {exec_path}")
    except Exception as save_exc:
        log(f"Failed to save executed notebook: {save_exc}")

    # Write log file
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    
    return {
        "notebook": nb_name,
        "backend": backend,
        "kernel": kernel,
        "interpreter": interpreter,
        "duration_sec": round(t_nb_duration, 2),
        "expected_configs": expected,
        "config_results": config_results,
        "error": nb_error,
    }


def main():
    print("=" * 80)
    print("DEEP HEDGING 7 NOTEBOOKS WINDOWS CPU SMOKE TEST RUNNER")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 80)

    # 1. Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Record environment
    capture_environment_info()

    # 3. Run all 7 notebooks sequentially
    all_nb_results = []
    t_global_start = time.time()

    for spec in NOTEBOOKS:
        nb_res = run_single_notebook(spec)
        all_nb_results.append(nb_res)

    t_global_duration = time.time() - t_global_start

    # 4. Synthesize 24 configurations summary
    summary_records = []
    for nb_res in all_nb_results:
        nb_name = nb_res["notebook"]
        backend = nb_res["backend"]
        interp = nb_res["interpreter"]
        expected_ids = nb_res["expected_configs"]
        cfg_results = nb_res["config_results"]
        nb_err = nb_res["error"]

        for cid in expected_ids:
            if cid in cfg_results:
                m = cfg_results[cid]
                samples = m.get("samples")
                epochs = m.get("epochs")
                train_l = m.get("train_loss")
                val_l = m.get("val_loss")
                is_fin = m.get("is_finite")
                dur = m.get("cell_duration_sec", 0.0)

                # Strict validation rule:
                # - samples == 500
                # - epochs == 2
                # - train_l is finite float
                # - val_l is finite float
                # - is_fin is True
                if (epochs == 2 and samples == 500 and 
                    train_l is not None and val_l is not None and 
                    is_fin):
                    status = "PASS"
                    err = ""
                else:
                    status = "FAIL"
                    err = f"Validation failed: epochs={epochs}, samples={samples}, finite={is_fin}"

                rec = {
                    "notebook": nb_name,
                    "config_id": cid,
                    "backend": backend,
                    "interpreter": interp,
                    "samples": samples,
                    "actual_epochs": epochs,
                    "elapsed_time_sec": dur,
                    "train_loss": train_l,
                    "val_loss": val_l,
                    "status": status,
                    "error": err
                }
            else:
                rec = {
                    "notebook": nb_name,
                    "config_id": cid,
                    "backend": backend,
                    "interpreter": interp,
                    "samples": None,
                    "actual_epochs": None,
                    "elapsed_time_sec": 0.0,
                    "train_loss": None,
                    "val_loss": None,
                    "status": "FAIL",
                    "error": f"Config not executed (notebook error: {nb_err})"
                }
            summary_records.append(rec)

    # 5. Save summary.json
    summary_json_path = RESULTS_DIR / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": TIMESTAMP,
            "results_dir": str(RESULTS_DIR),
            "total_duration_sec": round(t_global_duration, 2),
            "total_configs": len(summary_records),
            "pass_count": sum(1 for r in summary_records if r["status"] == "PASS"),
            "fail_count": sum(1 for r in summary_records if r["status"] != "PASS"),
            "configurations": summary_records,
            "notebooks": all_nb_results,
        }, f, indent=2)

    # 6. Save summary.csv
    summary_csv_path = RESULTS_DIR / "summary.csv"
    fieldnames = [
        "notebook", "config_id", "backend", "interpreter",
        "samples", "actual_epochs", "elapsed_time_sec",
        "train_loss", "val_loss", "status", "error"
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_records)

    # 7. Print final report
    pass_cnt = sum(1 for r in summary_records if r["status"] == "PASS")
    fail_cnt = sum(1 for r in summary_records if r["status"] != "PASS")

    print("\n" + "=" * 110)
    print("FINAL 24 CONFIGURATIONS SMOKE TEST RESULTS TABLE")
    print("=" * 110)
    print(f"{'#':<3} | {'Notebook':<38} | {'Config ID':<22} | {'Backend':<10} | {'Samples':<7} | {'Ep':<3} | {'Train Loss':<11} | {'Val Loss':<11} | {'Status':<6}")
    print("-" * 110)
    for i, r in enumerate(summary_records, 1):
        tl_str = f"{r['train_loss']:.6f}" if r['train_loss'] is not None else "N/A"
        vl_str = f"{r['val_loss']:.6f}" if r['val_loss'] is not None else "N/A"
        ep_str = str(r['actual_epochs']) if r['actual_epochs'] is not None else "N/A"
        sm_str = str(r['samples']) if r['samples'] is not None else "N/A"
        print(f"{i:<3} | {r['notebook']:<38} | {r['config_id']:<22} | {r['backend']:<10} | {sm_str:<7} | {ep_str:<3} | {tl_str:<11} | {vl_str:<11} | {r['status']:<6}")
    print("-" * 110)
    print(f"TOTAL: {len(summary_records)} configs | PASS: {pass_cnt} | FAIL: {fail_cnt} | Time: {t_global_duration:.2f}s")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 110)


if __name__ == "__main__":
    main()
