# B?ng ?nh X? 24 C?u H?nh Deep Hedging (Mapping Table)

B?ng ?nh x? chi ti?t 24 c?u h?nh benchmark Deep Hedging gi?a m? ngu?n g?c v? b? m? vi?t l?i ??c l?p.

## 1. Nguy?n T?c Thi?t K? & Ph?n B? Backend
- **JAX (`dh_jax`)**: ?p d?ng cho 4 notebook (14 c?u h?nh): `trainer.ipynb`, `blackscholes.ipynb`, `trainer-american_digital.ipynb`, `trainer-recurrent-barrier_smalldt.ipynb`.
- **TensorFlow (`dh_tf`)**: ?p d?ng cho 3 notebook (10 c?u h?nh): `trainer-bs.ipynb`, `trainer-recurrent-barrier.ipynb`, `trainer-recurrent-fwdstart.ipynb`.
- **??c l?p ho?n to?n**: Kh?ng s? d?ng package `deephedging` g?c; s? d?ng t??ng ?ng `dh_jax` ho?c `dh_tf` n?m tr?c ti?p trong th? m?c b?n giao.
- **H? tr? 2 ch? ??**:
  - `original`: Gi? nguy?n to?n b? 100% tham s? g?c (samples, epochs, batch_size, architecture, learning rate, clipping).
  - `smoke`: Gi?m s? m?u ($N \le 500$) v? s? epoch (2 epoch) ?? ki?m th? lu?ng th?c thi nhanh ch?ng m? kh?ng thay ??i ki?n tr?c m?ng, features, payoff hay h?m m?c ti?u.

---

## 2. B?ng ?nh X? Chi Ti?t 24 C?u H?nh

| STT | Notebook Vi?t L?i | T?n C?u H?nh (ID) | Notebook & Cell Ngu?n | Backend | Samples (Orig/Smoke) | Steps ($T$) | Step Size ($dt$) | Payoff | Ki?n Tr?c M?ng (Depth x Width, Act) | Tr?ng Th?i Recurrent | Utility & $\lambda$ | Optimizer, LR, Batch Size | Gradient Clipping | Epochs (Orig/Smoke) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `blackscholes.ipynb` | `bs_base` | `blackscholes.ipynb` [Cell 4] | **JAX** | 50,000 / 500 | 20 | 0.02 | ATM Call ($\mu=0.1, \sigma=0.2$) | $5 	imes 40$, Softplus | Kh?ng (none) | CVaR ($\lambda=1.0$) | Adam, $lpha=0.005$, batch=256 | Kh?ng | 400 / 2 |
| 2 | `blackscholes.ipynb` | `bs_exp2` | `blackscholes.ipynb` [Cell 9] | **JAX** | 50,000 / 500 | 20 | 0.02 | ATM Call ($\mu=0.0, \sigma=0.2$) | $5 	imes 40$, Softplus | Kh?ng (none) | Exp2 ($\lambda=10.0$) | Adam, $lpha=0.005$, batch=256 | Kh?ng | 400 / 2 |
| 3 | `blackscholes.ipynb` | `bs_nodrift` | `blackscholes.ipynb` [Cell 14] | **JAX** | 50,000 / 500 | 20 | 0.02 | ATM Call ($\mu=0.0, \sigma=0.2$) | $5 	imes 40$, Softplus | Kh?ng (none) | CVaR ($\lambda=1.0$) | Adam, $lpha=0.005$, batch=256 | `clipval=0.5, norm=1.0` | 600 / 2 |
| 4 | `blackscholes.ipynb` | `bs_nodrift_exp2` | `blackscholes.ipynb` [Cell 19] | **JAX** | 50,000 / 500 | 20 | 0.02 | ATM Call ($\mu=0.0, \sigma=0.2$) | $5 	imes 40$, Softplus | Kh?ng (none) | Exp2 ($\lambda=10.0$) | Adam, $lpha=0.001$, batch=256 | `clipval=0.5, norm=1.0` | 600 / 2 |
| 5 | `blackscholes.ipynb` | `bs_small` | `blackscholes.ipynb` [Cell 24] | **JAX** | 10,000 / 500 | 20 | 0.02 | ATM Call ($\mu=0.1, \sigma=0.2$) | $5 	imes 40$, Softplus | Kh?ng (none) | CVaR ($\lambda=2.0$) | Adam, $lpha=0.005$, batch=256 | Kh?ng | 10 / 2 |
| 6 | `trainer-bs.ipynb` | `trainer_bs_coarse` | `trainer-bs.ipynb` [Cell 2, 4] | **TensorFlow** | 10,000 / 500 | 20 | 0.02 | ATM Call ($\mu=0.0, \sigma=0.2$) | $3 	imes 20$, Softplus | Kh?ng (none) | CVaR ($\lambda=1.0$) | Adam, $lpha=0.005$, batch=2000 | Kh?ng | 20,000 / 2 |
| 7 | `trainer-bs.ipynb` | `trainer_bs_fine` | `trainer-bs.ipynb` [Cell 2, 7] | **TensorFlow** | 10,000 / 500 | 100 | 0.004 | ATM Call ($\mu=0.0, \sigma=0.2$) | $3 	imes 20$, Softplus | Kh?ng (none) | CVaR ($\lambda=1.0$) | Adam, $lpha=0.005$, batch=2000 | Kh?ng | 20,000 / 2 |
| 8 | `trainer.ipynb` | `trainer_bs` | `trainer.ipynb` [Cell 3] | **JAX** | 10,000 / 500 | 20 | 0.02 | ATM Call (Black-Scholes) | $3 	imes 20$, Softplus | Kh?ng (none) | CVaR ($\lambda=1.0$) | Adam, $lpha=0.005$, batch=32 | Kh?ng | 800 / 2 |
| 9 | `trainer.ipynb` | `trainer_sv` | `trainer.ipynb` [Cell 7] | **JAX** | 10,000 / 500 | 20 | 0.02 | ATM Call (Stoch Vol / Heston) | $3 	imes 20$, Softplus | Kh?ng (none) | CVaR ($\lambda=1.0$) | Adam, $lpha=0.001$, batch=32 | `clipval=1.0, norm=1.0` | 800 / 2 |
| 10 | `trainer-american_digital.ipynb` | `am_dig_none` | `trainer-american_digital.ipynb` [Cell 2, 4] | **JAX** | 20,000 / 500 | 100 | 0.02 | American Digital ($B=0.95$) | $5 	imes 50$, Softplus | `none` (0,0,0,0) | Exp2 ($\lambda=1.0$) | Adam, $lpha=0.0001$, batch=32 | `clipval=0.9, norm=1.0` | 500 / 2 |
| 11 | `trainer-american_digital.ipynb` | `am_dig_digital` | `trainer-american_digital.ipynb` [Cell 2, 4] | **JAX** | 20,000 / 500 | 100 | 0.02 | American Digital ($B=0.95$) | $5 	imes 50$, Softplus | `digital` (past=5, evt=5) | Exp2 ($\lambda=1.0$) | Adam, $lpha=0.0001$, batch=32 | `clipval=0.9, norm=1.0` | 500 / 2 |
| 12 | `trainer-american_digital.ipynb` | `am_dig_standard` | `trainer-american_digital.ipynb` [Cell 2, 4] | **JAX** | 20,000 / 500 | 100 | 0.02 | American Digital ($B=0.95$) | $5 	imes 50$, Softplus | `standard` (aggr=5) | Exp2 ($\lambda=1.0$) | Adam, $lpha=0.0001$, batch=32 | `clipval=0.9, norm=1.0` | 500 / 2 |
| 13 | `trainer-american_digital.ipynb` | `am_dig_mix_relu` | `trainer-american_digital.ipynb` [Cell 2, 4] | **JAX** | 20,000 / 500 | 100 | 0.02 | American Digital ($B=0.95$) | $5 	imes 100$, **ReLU** | `mix_relu` (aggr=2, past=2) | Exp2 ($\lambda=1.0$) | Adam, $lpha=0.0001$, batch=32 | `clipval=0.9, norm=1.0` | 500 / 2 |
| 14 | `trainer-american_digital.ipynb` | `am_dig_mix_softplus` | `trainer-american_digital.ipynb` [Cell 2, 4] | **JAX** | 20,000 / 500 | 100 | 0.02 | American Digital ($B=0.95$) | $5 	imes 100$, Softplus | `mix_softplus` (aggr=2, past=2) | Exp2 ($\lambda=1.0$) | Adam, $lpha=0.0001$, batch=32 | `clipval=0.9, norm=1.0` | 500 / 2 |
| 15 | `trainer-recurrent-barrier.ipynb` | `barrier_none` | `trainer-recurrent-barrier.ipynb` [Cell 2, 4] | **TensorFlow** | 20,000 / 500 | 100 | 0.004 | KO Put ($K=1.0, B=0.95$) | $5 	imes 50$, Softplus | `none` (0,0,0,0) | Exp ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 250 / 2 |
| 16 | `trainer-recurrent-barrier.ipynb` | `barrier_digital` | `trainer-recurrent-barrier.ipynb` [Cell 2, 4] | **TensorFlow** | 20,000 / 500 | 100 | 0.004 | KO Put ($K=1.0, B=0.95$) | $5 	imes 50$, Softplus | `digital` (past=5, evt=5) | Exp ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 250 / 2 |
| 17 | `trainer-recurrent-barrier_smalldt.ipynb` | `barrier_sdt_none` | `trainer-recurrent-barrier_smalldt.ipynb` [Cell 2, 4] | **JAX** | 20,000 / 500 | 20 | 0.02 | KO Put ($K=1.0, B=0.95$) | $5 	imes 50$, Softplus | `none` (0,0,0,0) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 250 / 2 |
| 18 | `trainer-recurrent-barrier_smalldt.ipynb` | `barrier_sdt_digital` | `trainer-recurrent-barrier_smalldt.ipynb` [Cell 2, 4] | **JAX** | 20,000 / 500 | 20 | 0.02 | KO Put ($K=1.0, B=0.95$) | $5 	imes 50$, Softplus | `digital` (past=5, evt=5) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 250 / 2 |
| 19 | `trainer-recurrent-fwdstart.ipynb` | `fwd_none` | `trainer-recurrent-fwdstart.ipynb` [Cell 2, 3] | **TensorFlow** | 20,000 / 500 | 20 | 0.004 | Fwd Put ($t_{reset}=10$) | $5 	imes 20$, Softplus | `none` (0,0,0,0) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 2000 / 2 |
| 20 | `trainer-recurrent-fwdstart.ipynb` | `fwd_all_modes` | `trainer-recurrent-fwdstart.ipynb` [Cell 2, 4] | **TensorFlow** | 20,000 / 500 | 20 | 0.004 | Fwd Put ($t_{reset}=10$) | $5 	imes 20$, Softplus | `all_modes` (cls=2, aggr=2, pst=2, evt=2) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 2000 / 2 |
| 21 | `trainer-recurrent-fwdstart.ipynb` | `fwd_aggr_2` | `trainer-recurrent-fwdstart.ipynb` [Cell 2, 5] | **TensorFlow** | 20,000 / 500 | 20 | 0.004 | Fwd Put ($t_{reset}=10$) | $5 	imes 20$, Softplus | `aggregate_2` (aggr=2) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 2000 / 2 |
| 22 | `trainer-recurrent-fwdstart.ipynb` | `fwd_aggr_5bnd` | `trainer-recurrent-fwdstart.ipynb` [Cell 2, 6] | **TensorFlow** | 20,000 / 500 | 20 | 0.004 | Fwd Put ($t_{reset}=10$) | $5 	imes 20$, Softplus | `aggregate_5bnd` (aggr=5, bound=True) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 2000 / 2 |
| 23 | `trainer-recurrent-fwdstart.ipynb` | `fwd_aggr_5` | `trainer-recurrent-fwdstart.ipynb` [Cell 2, 7] | **TensorFlow** | 20,000 / 500 | 20 | 0.004 | Fwd Put ($t_{reset}=10$) | $5 	imes 20$, Softplus | `aggregate_5` (aggr=5, bound=False) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 2000 / 2 |
| 24 | `trainer-recurrent-fwdstart.ipynb` | `fwd_past_repr` | `trainer-recurrent-fwdstart.ipynb` [Cell 2, 8] | **TensorFlow** | 20,000 / 500 | 20 | 0.004 | Fwd Put ($t_{reset}=10$) | $5 	imes 20$, Softplus | `past_repr` (past=2) | CVaR ($\lambda=1.01$) | Adam, $lpha=0.001$, batch=32 | `clipval=0.9, norm=1.0` | 2000 / 2 |

---

## 3. Kh?c Bi?t Gi?a Hai B? M? Ngu?n & C?ch X? L?

1. **Look-ahead Bias trong Barrier Payoff:**
   - *Kh?c bi?t:* Trong m?t s? tri?n khai m?u c?, h?m payoff ki?m tra ?i?u ki?n ch?m r?o qua `spots > barrier` kh?ng t?nh tu?n t? nh?n qu? (causality), d?n ??n r? r? th?ng tin t??ng lai v? tr?ng th?i r?o c?n.
   - *C?ch x? l?:* S? d?ng `notko = np.cumprod(np.where(spots > barrier, 1.0, 0.0), axis=1)`. Gi? tr? r?o c?n t?i b??c $t$ ho?n to?n ph? thu?c v?o qu? kh? v? hi?n t?i.
2. **K?ch th??c chi?u th?i gian c?a Payoff Features (`features` vs `spot_all`):**
   - *Kh?c bi?t:* M?ng `spots` ch?a $N+1$ b??c th?i gian ($0, \dots, T$). Tuy nhi?n, h?nh ??ng ph?ng h? di?n ra t?i $T$ th?i ?i?m ($0, \dots, T-1$).
   - *C?ch x? l?:* C?t l?t ??c tr?ng r?o c?n th?nh `notko[:, :-1]` c? ??ng ?? d?i $T$, tr?nh l?i k?ch th??c trong `jax.lax.scan` v? Keras RNN loop.
3. **Typo `bound_aggr_states`:**
   - *Kh?c bi?t:* Thu?c t?nh kh?ng ch? tr?ng th?i b? nh? t?ch l?y trong kho?ng $(-1, 1)$ b? g?n l?p ho?c sai t?n trong m?t s? notebook g?c.
   - *C?ch x? l?:* ??a v?o c?u h?nh chu?n h?a r? r?ng: `bound_aggr_states = True` trong c?u h?nh `fwd_aggr_5bnd` v? `False` trong `fwd_aggr_5`.
4. **Xung ??t ??nh danh th? vi?n (`deephedging` namespace collision):**
   - *Kh?c bi?t:* C? repo TF v? repo JAX ??u d?ng t?n `deephedging`, d?n t?i nguy c? n?p nh?m package khi chuy?n ??i gi?a hai backend ho?c khi ch?y tr?n m?y Windows.
   - *C?ch x? l?:* T?ch th?nh hai package con ??c l?p ho?n to?n: `dh_jax` (cho JAX) v? `dh_tf` (cho TensorFlow). C?c notebook import tr?c ti?p t? local package m? kh?ng c?n c?i ??t ho?c ph? thu?c v?o th? m?c cluster.
