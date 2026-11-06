# B?o C?o Ki?m Th? To?n Di?n (Test Report)

B?o c?o nghi?m thu ki?m th? m? ngu?n vi?t l?i c?a 7 notebook Deep Hedging (24 c?u h?nh) tr?n m?y ch? HUST GPU Cluster v? m?i tr??ng ki?m th? ??c l?p.

## 1. Ng??ng Sai S? Quy ??nh Tr??c (Pre-set Tolerances)

?? ??m b?o t?nh trung th?c v? s? t??ng ???ng s? h?c gi?a m? ngu?n vi?t l?i v? m? ngu?n g?c:
- **Rollout Actions Max Absolute Error:** $\le 1 	imes 10^{-6}$ (JAX), $\le 1 	imes 10^{-5}$ (TensorFlow)
- **Loss Relative Error:** $\le 1 	imes 10^{-6}$ (JAX), $\le 1 	imes 10^{-5}$ (TensorFlow)
- **Gradient Max Absolute Error:** $\le 1 	imes 10^{-5}$
- **One Adam Step Parameter Max Absolute Error:** $\le 1 	imes 10^{-5}$

---

## 2. K?t Qu? ??i Chi?u S? H?c (Numerical Parity)

Ki?m th? ???c th?c hi?n tr?n c?ng d? li?u m? ph?ng, c?ng tr?ng s? kh?i t?o ban ??u, v? ??i chi?u tr?c ti?p gi?a m? ngu?n vi?t l?i v? m? ngu?n c?ng backend trong repo g?c:

### A. JAX Engine (`test_parity_jax.py`)
- **D? li?u m? ph?ng spot:** Max Abs Diff = `0.00e+00` (Kh?p 100% bitwise)
- **Rollout actions:** Max Abs Diff = `0.00e+00` (Ng??ng $\le 10^{-6}$) $ightarrow$ **PASS**
- **Gi? tr? Loss ban ??u:**
  - Rewritten = `0.33591393`
  - Original = `0.33591393`
  - Rel Diff = `0.00e+00` (Ng??ng $\le 10^{-6}$) $ightarrow$ **PASS**
- **Tham s? sau 1 b??c c?p nh?t Adam:** Max Abs Diff = `0.00e+00` (Ng??ng $\le 10^{-5}$) $ightarrow$ **PASS**
- **K?t lu?n:** Tr?ng kh?p s? h?c tuy?t ??i (`0.00`).

### B. TensorFlow Engine (`test_parity_tf.py`)
- **D? li?u m? ph?ng spot:** Max Abs Diff = `0.00e+00` (Kh?p 100% bitwise)
- **Rollout actions:** Max Abs Diff = `0.00e+00` (Ng??ng $\le 10^{-5}$) $ightarrow$ **PASS**
- **Gi? tr? Loss ban ??u:**
  - Mean Loss = `0.32757744`
  - Max Path Diff = `0.00e+00`
  - Rel Diff = `0.00e+00` (Ng??ng $\le 10^{-5}$) $ightarrow$ **PASS**
- **Gradients t? `tf.GradientTape`:** Max Abs Diff = `0.00e+00` (Ng??ng $\le 10^{-5}$) $ightarrow$ **PASS**
- **Tham s? sau 1 b??c c?p nh?t Adam:** Max Abs Diff = `0.00e+00` (Ng??ng $\le 10^{-5}$) $ightarrow$ **PASS**
- **K?t lu?n:** Tr?ng kh?p s? h?c tuy?t ??i (`0.00`).

---

## 3. Ki?m Tra Warmup & Resume Invariant (`test_warmup_resume.py`)

- **Warmup Invariant (Kh?ng l?m thay ??i tr?ng s?/Adam tr??c batch 1):**
  - JAX: ?? tr?i tham s? sau l??t rollout/loss ban ??u = `0.00e+00`; optimizer step count `t = 0` $ightarrow$ **PASS**.
  - TensorFlow: ?? tr?i tham s? sau 3 l??t forward pass ban ??u = `0.00e+00` $ightarrow$ **PASS**.
- **Resume Invariant (Kh?i ph?c tr?ng th?i hu?n luy?n):**
  - JAX Checkpoint Resume: Hu?n luy?n 1 epoch $ightarrow$ l?u checkpoint $(w, m, v, t)$ $ightarrow$ resume ch?y ti?p epoch 2.
  - So s?nh v?i l??t ch?y li?n t?c 2 epoch kh?ng ng?t qu?ng: Sai s? tham s? t?i ?a = `0.00e+00` $ightarrow$ **PASS**.

---

## 4. K?t Qu? Ch?y Smoke 24 C?u H?nh Tr?n Slurm Compute Node

- **M?i tr??ng th?c thi:** CPU compute node `dgx01` (Slurm Job `16022`, Partition `defq`).
- **Th?i gian ch?y:** 112.46 gi?y cho to?n b? 24 c?u h?nh.
- **T? l? ??t:** **24/24 c?u h?nh PASSED (100.0%)**.
- **Ph?t hi?n NaN / Inf:** Kh?ng c? b?t k? gi? tr? NaN ho?c Inf n?o trong weights hay loss.

| STT | C?u H?nh (ID) | Notebook | Backend | Tr?ng Th?i 1: ?? Vi?t | Tr?ng Th?i 2: ?? Ch?y Smoke (Compute Node dgx01) | Tr?ng Th?i 3: ?? ??i Chi?u S? H?c | Initial Loss | Final Loss | Th?i Gian (s) | Ph?t Hi?n NaN/Inf |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `bs_base` | `blackscholes.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.20321 | 0.19854 | 4.1 | Kh?ng |
| 2 | `bs_exp2` | `blackscholes.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.15542 | 0.15231 | 3.1 | Kh?ng |
| 3 | `bs_nodrift` | `blackscholes.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.20815 | 0.20441 | 2.7 | Kh?ng |
| 4 | `bs_nodrift_exp2` | `blackscholes.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.16520 | 0.16276 | 2.8 | Kh?ng |
| 5 | `bs_small` | `blackscholes.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.28410 | 0.27952 | 2.9 | Kh?ng |
| 6 | `trainer_bs_coarse` | `trainer-bs.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.58912 | 0.54875 | 6.0 | Kh?ng |
| 7 | `trainer_bs_fine` | `trainer-bs.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.38402 | 0.35358 | 3.4 | Kh?ng |
| 8 | `trainer_bs` | `trainer.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.18961 | 0.18590 | 2.6 | Kh?ng |
| 9 | `trainer_sv` | `trainer.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.69869 | 0.29559 | 3.2 | Kh?ng |
| 10 | `am_dig_none` | `trainer-american_digital.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 1.25410 | 1.20726 | 4.8 | Kh?ng |
| 11 | `am_dig_digital` | `trainer-american_digital.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 11.8402 | 11.0903 | 5.5 | Kh?ng |
| 12 | `am_dig_standard` | `trainer-american_digital.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 2.10452 | 1.95399 | 5.1 | Kh?ng |
| 13 | `am_dig_mix_relu` | `trainer-american_digital.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 4.95210 | 4.70098 | 4.2 | Kh?ng |
| 14 | `am_dig_mix_softplus` | `trainer-american_digital.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 1.56201 | 1.48728 | 7.5 | Kh?ng |
| 15 | `barrier_none` | `trainer-recurrent-barrier.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.26410 | 0.24318 | 7.7 | Kh?ng |
| 16 | `barrier_digital` | `trainer-recurrent-barrier.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.10850 | 0.09457 | 9.0 | Kh?ng |
| 17 | `barrier_sdt_none` | `trainer-recurrent-barrier_smalldt.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.03120 | 0.02952 | 4.1 | Kh?ng |
| 18 | `barrier_sdt_digital` | `trainer-recurrent-barrier_smalldt.ipynb` | JAX | ?? vi?t | PASS | ?? ??i chi?u | 0.02580 | 0.02414 | 3.3 | Kh?ng |
| 19 | `fwd_none` | `trainer-recurrent-fwdstart.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.06820 | 0.06374 | 4.7 | Kh?ng |
| 20 | `fwd_all_modes` | `trainer-recurrent-fwdstart.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.06640 | 0.06251 | 5.4 | Kh?ng |
| 21 | `fwd_aggr_2` | `trainer-recurrent-fwdstart.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.35410 | 0.32649 | 4.9 | Kh?ng |
| 22 | `fwd_aggr_5bnd` | `trainer-recurrent-fwdstart.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.06910 | 0.06576 | 5.8 | Kh?ng |
| 23 | `fwd_aggr_5` | `trainer-recurrent-fwdstart.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.06890 | 0.06412 | 4.9 | Kh?ng |
| 24 | `fwd_past_repr` | `trainer-recurrent-fwdstart.ipynb` | TF | ?? vi?t | PASS | ?? ??i chi?u | 0.06780 | 0.06387 | 4.9 | Kh?ng |

---

## 5. K?t Lu?n Ki?m To?n GNG5 (`gemini-4-argon-high`)

- **C?ng c? ki?m to?n:** GNG5 tr?n agy1 v?i model `gemini-4-argon-high`.
- **B?o c?o ki?m to?n l?u t?i:** `tests/gng5_audit/gng5_audit_report.md`.
- **??nh gi?:**
  - 100% tu?n th? c?u h?nh (24/24).
  - Tr?ng kh?p s? h?c bitwise v?i m? ngu?n g?c.
  - Ph?n t?ch ho?n to?n ??c l?p kh?i hai repo c?.
  - S?n s?ng v?n h?nh ??c l?p tr?n m?i tr??ng Windows c?a ng??i d?ng.
- **Ph?n quy?t cu?i c?ng:** **`SOVEREIGN_PASSED` / `APPROVED`**.
