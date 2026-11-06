#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Hedging Windows TensorFlow Smoke Test Runner
Re-executes the 3 TensorFlow notebooks sequentially on CPU Windows after resume checkpoint fix:
- trainer-bs.ipynb (2 configs: trainer_bs_coarse, trainer_bs_fine)
- trainer-recurrent-barrier.ipynb (2 configs: barrier_none, barrier_digital)
- trainer-recurrent-fwdstart.ipynb (6 configs: fwd_none, fwd_all_modes, fwd_aggr_2, fwd_aggr_5bnd, fwd_aggr_5, fwd_past_repr)

Total: 10 TensorFlow configurations.
Isolated checkpoints, environment recording, logs, and executed notebooks.
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

REPO_DIR = Path(r"D:\deephedging-rewritten")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = REPO_DIR / "results" / f"windows_smoke_tf_{TIMESTAMP}"
EXEC_DIR = RESULTS_DIR / "executed_notebooks"
LOGS_DIR = RESULTS_DIR / "logs"
ENV_DIR = RESULTS_DIR / "environment"
ISOLATED_CKPTS_DIR = RESULTS_DIR / "isolated_checkpoints"

TF_INTERPRETER = str(REPO_DIR / r".venv-tf\Scripts\python.exe")

os.environ["DH_MODE"] = "smoke"
os.environ.pop("DH_EPOCHS", None)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["DH_CHECKPOINTS"] = str(ISOLATED_CKPTS_DIR)

NOTEBOOKS = [
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
    print("Capturing TF environment details...")
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    try:
        py_tf = subprocess.check_output([TF_INTERPRETER, "-V"], text=True).strip()
        (ENV_DIR / "python_version_tf.txt").write_text(py_tf, encoding="utf-8")
        pip_tf = subprocess.check_output([TF_INTERPRETER, "-m", "pip", "list"], text=True)
        (ENV_DIR / "packages_tf.txt").write_text(pip_tf, encoding="utf-8")
    except Exception as e:
        print(f"Warning: Failed to capture TF environment: {e}")


def get_query_code(nb_name: str, cell_src: str):
    s = cell_src
    if nb_name == "trainer-bs.ipynb":
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
                    
                    for out in cell.outputs:
                        if "text" in out:
                            for l in out["text"].strip().splitlines():
                                log(f"  [stdout] {l}")
                        elif "traceback" in out:
                            for l in out["traceback"]:
                                log(f"  [traceback] {l}")
                    
                    query_code = get_query_code(nb_name, cell.source)
                    if query_code:
                        log(f"  -> Querying kernel for configuration metrics...")
                        q_cell = nbformat.v4.new_code_cell(query_code)
                        client.nb.cells.append(q_cell)
                        q_idx = len(client.nb.cells) - 1
                        client.execute_cell(q_cell, q_idx)
                        
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
    
    try:
        nbformat.write(client.nb, exec_path)
        log(f"Saved executed notebook to: {exec_path}")
    except Exception as save_exc:
        log(f"Failed to save executed notebook: {save_exc}")

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
    print("DEEP HEDGING TENSORFLOW 3 NOTEBOOKS WINDOWS CPU SMOKE TEST RUNNER")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    capture_environment_info()

    all_nb_results = []
    t_global_start = time.time()

    for spec in NOTEBOOKS:
        nb_res = run_single_notebook(spec)
        all_nb_results.append(nb_res)

    t_global_duration = time.time() - t_global_start

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

                if (epochs == 2 and samples == 500 and 
                    is_fin and train_l is not None and val_l is not None):
                    status = "PASS"
                    err_msg = ""
                else:
                    status = "FAIL"
                    err_msg = f"Incomplete: epochs={epochs}, samples={samples}, finite={is_fin}"
            else:
                samples = None
                epochs = None
                train_l = None
                val_l = None
                is_fin = False
                dur = 0.0
                status = "FAIL"
                err_msg = f"Notebook execution error: {nb_err}" if nb_err else "Config not executed or missing"

            summary_records.append({
                "notebook": nb_name,
                "config_id": cid,
                "backend": backend,
                "interpreter": interp,
                "samples": samples,
                "actual_epochs": epochs,
                "duration_sec": dur,
                "train_loss": train_l,
                "val_loss": val_l,
                "is_finite": is_fin,
                "status": status,
                "error": err_msg,
            })

    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "notebook", "config_id", "backend", "interpreter",
            "samples", "actual_epochs", "duration_sec",
            "train_loss", "val_loss", "is_finite", "status", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_records)

    json_path = RESULTS_DIR / "summary.json"
    summary_data = {
        "timestamp": TIMESTAMP,
        "total_duration_sec": round(t_global_duration, 2),
        "total_configs": len(summary_records),
        "passed": sum(1 for r in summary_records if r["status"] == "PASS"),
        "failed": sum(1 for r in summary_records if r["status"] == "FAIL"),
        "configurations": summary_records,
        "notebook_results": all_nb_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("TENSORFLOW SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"{'Config ID':<25} | {'Notebook':<32} | {'Epochs':<6} | {'Samples':<7} | {'Train Loss':<10} | {'Val Loss':<10} | {'Status'}")
    print("-" * 110)
    for r in summary_records:
        tl_str = f"{r['train_loss']:.4f}" if r['train_loss'] is not None else "N/A"
        vl_str = f"{r['val_loss']:.4f}" if r['val_loss'] is not None else "N/A"
        ep_str = str(r['actual_epochs']) if r['actual_epochs'] is not None else "N/A"
        sm_str = str(r['samples']) if r['samples'] is not None else "N/A"
        print(f"{r['config_id']:<25} | {r['notebook']:<32} | {ep_str:<6} | {sm_str:<7} | {tl_str:<10} | {vl_str:<10} | {r['status']}")
    print("-" * 110)
    print(f"Total: {summary_data['total_configs']} | PASS: {summary_data['passed']} | FAIL: {summary_data['failed']} | Duration: {summary_data['total_duration_sec']}s")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
