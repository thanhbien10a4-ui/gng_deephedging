# BÁO CÁO KIỂM THỬ SỐ HỌC VÀ ĐỐI CHIẾU MÔ HÌNH (NUMERICAL VERIFICATION REPORT)
**Thư mục lưu trữ**: `results/windows_verification_20261106_165611/`  
**Thời gian thực hiện**: 2026-11-06 17:10:00 (UTC+7)  
**Môi trường thực thi**: Windows 11 CPU  
- **JAX**: Python 3.13.0 (`C:\Users\Hi Windows 11 Home\AppData\Local\Programs\Python\Python313\python.exe`)
- **TensorFlow**: Python 3.10.11 VirtualEnv (`D:\deephedging-rewritten\.venv-tf\Scripts\python.exe`)

---

## 1. TỔNG HỢP KẾT QUẢ KIỂM THỬ (VERDICT SUMMARY)

| Hạng mục kiểm tra | Backend | Trạng thái | Bằng chứng định lượng / Sai lệch thực tế |
| :--- | :--- | :---: | :--- |
| **Parity Test (Rollout, Loss, Gradients, Adam)** | **JAX** | **PASS** | Spot diff: `0.00e+00`, Rollout diff: `0.00e+00`, RelLoss diff: `0.00e+00`, Adam param diff: `0.00e+00` |
| **Parity Test (Rollout, Loss, Gradients, Adam)** | **TensorFlow** | **PASS** | Spot diff: `0.00e+00`, Rollout diff: `0.00e+00`, RelLoss diff: `0.00e+00`, Adam param diff: `0.00e+00` |
| **Warmup Invariant** | **JAX** | **PASS** | Weight drift: `0.00e+00`, Optimizer step: `0` (không biến đổi trọng số/optimizer) |
| **Warmup Invariant** | **TensorFlow** | **PASS** | Weight drift qua 3 forward pass: `0.00e+00` |
| **Resume Invariant (Tiến trình mới)** | **JAX** | **PASS** | Weight diff vs liên tục: `0.00e+00`, Epoch 2 loss diff: `0.00e+00` (khôi phục hoàn hảo `carry=(params, opt)`) |
| **Resume Invariant (Tiến trình mới)** | **TensorFlow** | **FAIL** | Weight diff vs liên tục: `2.04e-03` (> ngưỡng `1e-5`). Bug mã nguồn: `restore_from_cache` return sớm bỏ qua nạp optimizer |

---

## 2. KIỂM TRA TÍNH HỢP LỆ VÀ PHẠM VI CỦA PARITY TESTS

### 2.1. Xác minh cách ly mã nguồn (No Self-Import)
- **JAX**:
  - Mã viết lại: `dh_jax` tại `D:\deephedging-rewritten\dh_jax\model.py`.
  - Mã tham chiếu: `orig_jax` nạp động qua `SourceFileLoader` từ `D:\deephedging-jax\deephedging\model.py`.
  - Hai implementation hoàn toàn độc lập, khác biệt đường dẫn vật lý và namespace import.
- **TensorFlow**:
  - Mã viết lại: `dh_tf` tại `D:\deephedging-rewritten\dh_tf\gym.py`.
  - Mã tham chiếu: `deephedging` tại `D:\deephedging\gym.py`.
  - Hai implementation hoàn toàn độc lập trên đĩa và trong module loader.

### 2.2. Chi tiết sai số kiểm thử Parity
Cả hai backend đều dùng chung seed dữ liệu, chung đường đi giá giao ngay ($S_t$) và đồng bộ trọng số khởi tạo 100%:

| Tham số / Đại lượng | JAX (`test_parity_jax.py`) | TensorFlow (`test_parity_tf.py`) | Ngưỡng cho phép |
| :--- | :---: | :---: | :---: |
| **Spot Path Max Abs Diff** | `0.00e+00` | `0.00e+00` | $0.0$ |
| **Rollout Actions Max Abs Diff** | `0.00e+00` | `0.00e+00` | $\le 1 \times 10^{-6}$ (JAX) / $1 \times 10^{-5}$ (TF) |
| **Initial Loss (Rewritten vs Orig)** | `1.09503448` vs `1.09503448` | `0.27831483` vs `0.27831483` | — |
| **Initial Loss Rel Diff** | `0.00e+00` | `0.00e+00` | $\le 1 \times 10^{-6}$ (JAX) / $1 \times 10^{-5}$ (TF) |
| **Gradients Max Abs Diff** | `0.00e+00` | `0.00e+00` | $\le 1 \times 10^{-5}$ |
| **1 Adam Step Param Max Abs Diff** | `0.00e+00` | `0.00e+00` | $\le 1 \times 10^{-5}$ |

### 2.3. Minh bạch phạm vi cấu hình kiểm thử (Config Coverage)
> [!IMPORTANT]
> **Không suy rộng kết quả PASS cho cả 24 cấu hình**.
> - **Cấu hình JAX thực sự được kiểm thử parity**: 1 cấu hình duy nhất: Black-Scholes ATM Call feedforward (`samples=500, steps=20, dt=0.02, rvol=0.2, drift=0.1, depth=3, width=20, softplus, cvar, lmbda=1.0`).
>   *13 cấu hình JAX còn lại* (American Digital, Barrier Small-dt, Stochastic Volatility, Exponential Utility) **chưa có test parity độc lập trong `test_parity_jax.py`** -> Ghi nhận: **NOT TESTED IN PARITY SUITE**.
> - **Cấu hình TensorFlow thực sự được kiểm thử parity**: 1 cấu hình duy nhất: Black-Scholes ATM Call feedforward (`samples=200, steps=10, dt=0.02, drift=0.0, depth=3, width=20, softplus, cvar, lmbda=1.0`).
>   *9 cấu hình TF còn lại* (Fine dt 100 bước, Barrier KO put, Forward Start 6 biến thể) **chưa có test parity độc lập trong `test_parity_tf.py`** -> Ghi nhận: **NOT TESTED IN PARITY SUITE**.

---

## 3. KIỂM TRA WARMUP VÀ CHECKPOINT RESUME

### 3.1. Warmup Invariant
- **JAX**:
  - Chạy `rollout(p0, gym, data)` và `loss(p0, gym, data)`.
  - Sai lệch trọng số trước và sau warmup: `0.00e+00`.
  - Bước cập nhật optimizer Adam: $t = 0$.
  - **Kết luận**: **PASS**.
- **TensorFlow**:
  - Chạy 3 lượt forward pass `gym(world.tf_data)`.
  - Sai lệch trọng số trước và sau warmup: `0.00e+00`.
  - **Kết luận**: **PASS**.

### 3.2. Resume Invariant (So sánh chạy liên tục 2 epoch vs Nạp checkpoint trong tiến trình mới)
Quy trình kiểm thử:
1. Chạy liên tục 2 epoch với dữ liệu và thứ tự batch xác định.
2. Chạy 1 epoch, lưu checkpoint ra đĩa.
3. Mở một **tiến trình Python mới hoàn toàn** (`subprocess.run`), nạp checkpoint từ đĩa, chạy tiếp epoch 2.
4. Đối chiếu toàn bộ mảng trọng số, trạng thái Adam, số bước và hàm loss.

#### Kết quả JAX:
- Cơ chế JAX: Lưu trực tiếp `carry = (params, opt_state)` vào file pickle checkpoint; dùng `np.random.default_rng([c.seed, epoch])` để đảm bảo phân vị batch epoch 2 hoàn toàn trùng khớp.
- Sai lệch trọng số `best_params`: `0.00e+00` (Ngưỡng $\le 10^{-6}$).
- Sai lệch trọng số `last_params`: `0.00e+00`.
- Sai lệch Train Loss epoch 2: `0.00e+00`.
- Sai lệch Val Loss epoch 2: `0.00e+00`.
- Best epoch match: Epoch 1 (trùng khớp hoàn hảo).
- **Kết luận JAX Resume**: **PASS**.

#### Kết quả TensorFlow:
- Sai lệch trọng số lớn nhất giữa chạy liên tục và nạp tiến trình mới: **`2.0379e-03`** (Vượt ngưỡng cho phép $1 \times 10^{-5}$).
- **Nguyên nhân kỹ thuật chính xác**:
  1. Trong `dh_tf/gym.py` (cũng như mã gốc `deephedging/gym.py` dòng 367-371):
     ```python
     try:
         self.set_weights( gym_weights )
     except ValueError as v:
         _log.warn(...)
         return False
     return True  # <-- DÒNG 371: RETURN SỚM!
     
     if self.optimizer is None:
         return True
     ...
     self.optimizer.set_weights( opt_weights ) # <-- KHÔNG BAO GIỜ ĐƯỢC GỌI!
     ```
     Lệnh `return True` ở dòng 371 ngắt hàm ngay sau khi nạp weights của gym, khiến toàn bộ logic nạp trọng số optimizer Adam (`opt_weights`: momentum $m, v$) bên dưới bị bỏ qua.
  2. Khi nạp lại mô hình trong tiến trình mới, optimizer Adam bị khởi tạo mới tinh ($m=0, v=0, t=0$), mất toàn bộ gia tốc và thông tin gradient của epoch 1.
  3. Keras `Model.fit` không đồng bộ thứ tự shuffle giữa 2 lần gọi hàm độc lập nếu không truyền cố định batch generator.
- Tuân thủ quy định *"Không sửa công thức mô hình hoặc nới ngưỡng sai số để ép PASS"*:
- **Kết luận TF Resume**: **FAIL**.

---

## 4. RÀ SOÁT VÀ HIỆU CHỈNH BÁO CÁO SMOKE TEST (TỪ LOG GỐC)

### 4.1. Phân định thời gian nhóm (Cell Group Time) vs Thời gian từng cấu hình
Trong báo cáo `summary.csv` của smoke test (`results/windows_smoke_20261106_163915`), các notebook chạy vòng lặp trong 1 cell đã ghi nhận thời gian của cả ô lệnh cho từng cấu hình con. Cụ thể:

#### American Digital (`trainer-american_digital.ipynb`)
- Báo cáo cũ `summary.csv` ghi: `37.25s` cho mỗi cấu hình trong số 5 cấu hình.
- **Xác minh thực tế từ log `trainer-american_digital.log`**: `37.25s` là **thời gian nhóm của cả Cell 5**.
- Thời gian training thực tế từng cấu hình được bóc tách từ log epoch:
  - `am_dig_none`: **2.51s** (epoch 1: 2.06s, epoch 2: 0.45s)
  - `am_dig_digital`: **2.50s** (epoch 1: 2.08s, epoch 2: 0.42s)
  - `am_dig_standard`: **2.34s** (epoch 1: 1.94s, epoch 2: 0.40s)
  - `am_dig_mix_relu`: **2.51s** (epoch 1: 2.00s, epoch 2: 0.51s)
  - `am_dig_mix_softplus`: **3.11s** (epoch 1: 2.23s, epoch 2: 0.88s)
  - *Tổng thời gian train 5 cấu hình*: 12.97s (phần thời gian còn lại của Cell 5 là biên dịch JIT và khởi tạo).

#### Small-dt Recurrent Barrier (`trainer-recurrent-barrier_smalldt.ipynb`)
- Báo cáo cũ `summary.csv` ghi: `13.76s` cho mỗi cấu hình trong số 2 cấu hình.
- **Xác minh thực tế từ log `trainer-recurrent-barrier_smalldt.log`**: `13.76s` là **thời gian nhóm của cả Cell 5**.
- Thời gian training thực tế từng cấu hình được bóc tách từ log epoch:
  - `barrier_sdt_none`: **1.79s** (epoch 1: 1.70s, epoch 2: 0.09s)
  - `barrier_sdt_digital`: **2.17s** (epoch 1: 2.07s, epoch 2: 0.10s)
  - *Tổng thời gian train 2 cấu hình*: 3.96s (phần còn lại là biên dịch JIT đồ thị tuần hoàn).

### 4.2. Số mẫu Validation thực tế theo 24 cấu hình
Quy tắc sinh tập validation trong mã nguồn:
- JAX (`trainer.ipynb`, `blackscholes.ipynb`): `samples = max(n_samples // 10, 50)`. Với $N=500 \implies N_{val} = 50$.
- JAX (`trainer-american_digital.ipynb`, `trainer-recurrent-barrier_smalldt.ipynb`): `samples = max(n_samples // 2, 50)`. Với $N=500 \implies N_{val} = 250$.
- TF (`trainer-bs.ipynb`, `trainer-recurrent-barrier.ipynb`, `trainer-recurrent-fwdstart.ipynb`): `samples = max(n_samples // 2, 50)`. Với $N=500 \implies N_{val} = 250$.

### 4.3. Phân biệt Validation Set vs Tập Test Độc Lập (Out-of-sample Test Set)
- **Validation Set (`val_world`)**: Được mô phỏng song song qua `world.clone(...)`, dùng để tính hàm loss kiểm định sau mỗi epoch nhằm chọn `best_weights`. Toàn bộ 24 cấu hình đều có $N_{val}$ hợp lệ hữu hạn trong smoke test.
- **Tập Test Độc Lập (`test_samples`)**: Trong `catalog.py`, mỗi cấu hình định nghĩa số mẫu test out-of-sample riêng (từ 1,000 đến 10,000 mẫu). Trong smoke test, các notebook không chạy bước đánh giá out-of-sample này sau khi huấn luyện.
- Tuân thủ quy định *"số liệu không có thì ghi 'chưa đo'"*:
  - **Chỉ số Out-of-sample Test Loss của 24 cấu hình trong lượt chạy smoke**: **Chưa đo**.

---

## 5. BẢNG TỔNG HỢP CHI TIẾT 24 CẤU HÌNH (AUDIT MATRIX)

| STT | Notebook | Cấu hình ID | Backend | Mẫu Train ($N$) | Mẫu Val ($N_{val}$) | Mẫu Test độc lập ($N_{test}$) | Epoch thực tế | Thời gian ghi nhận | Phân loại thời gian | Parity Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 1 | trainer.ipynb | trainer_bs | JAX | 500 | 50 | Chưa đo | 2 | 7.07s | Cấu hình riêng | NOT TESTED |
| 2 | trainer.ipynb | trainer_sv | JAX | 500 | 50 | Chưa đo | 2 | 4.91s | Cấu hình riêng | NOT TESTED |
| 3 | blackscholes.ipynb | bs_base | JAX | 500 | 50 | Chưa đo | 2 | 9.95s | Cấu hình riêng | **PASS** (1/1) |
| 4 | blackscholes.ipynb | bs_exp2 | JAX | 500 | 50 | Chưa đo | 2 | 8.81s | Cấu hình riêng | NOT TESTED |
| 5 | blackscholes.ipynb | bs_nodrift | JAX | 500 | 50 | Chưa đo | 2 | 5.69s | Cấu hình riêng | NOT TESTED |
| 6 | blackscholes.ipynb | bs_nodrift_exp2 | JAX | 500 | 50 | Chưa đo | 2 | 7.72s | Cấu hình riêng | NOT TESTED |
| 7 | blackscholes.ipynb | bs_small | JAX | 500 | 50 | Chưa đo | 2 | 8.03s | Cấu hình riêng | NOT TESTED |
| 8 | trainer-american_digital.ipynb | am_dig_none | JAX | 500 | 250 | Chưa đo | 2 | 37.25s (train: 2.51s) | Nhóm Cell 5 | NOT TESTED |
| 9 | trainer-american_digital.ipynb | am_dig_digital | JAX | 500 | 250 | Chưa đo | 2 | 37.25s (train: 2.50s) | Nhóm Cell 5 | NOT TESTED |
| 10 | trainer-american_digital.ipynb | am_dig_standard | JAX | 500 | 250 | Chưa đo | 2 | 37.25s (train: 2.34s) | Nhóm Cell 5 | NOT TESTED |
| 11 | trainer-american_digital.ipynb | am_dig_mix_relu | JAX | 500 | 250 | Chưa đo | 2 | 37.25s (train: 2.51s) | Nhóm Cell 5 | NOT TESTED |
| 12 | trainer-american_digital.ipynb | am_dig_mix_softplus | JAX | 500 | 250 | Chưa đo | 2 | 37.25s (train: 3.11s) | Nhóm Cell 5 | NOT TESTED |
| 13 | trainer-recurrent-barrier_smalldt.ipynb | barrier_sdt_none | JAX | 500 | 250 | Chưa đo | 2 | 13.76s (train: 1.79s) | Nhóm Cell 5 | NOT TESTED |
| 14 | trainer-recurrent-barrier_smalldt.ipynb | barrier_sdt_digital | JAX | 500 | 250 | Chưa đo | 2 | 13.76s (train: 2.17s) | Nhóm Cell 5 | NOT TESTED |
| 15 | trainer-bs.ipynb | trainer_bs_coarse | TF | 500 | 250 | Chưa đo | 2 | 11.17s | Cấu hình riêng | **PASS** (1/1) |
| 16 | trainer-bs.ipynb | trainer_bs_fine | TF | 500 | 250 | Chưa đo | 2 | 5.77s | Cấu hình riêng | NOT TESTED |
| 17 | trainer-recurrent-barrier.ipynb | barrier_none | TF | 500 | 250 | Chưa đo | 2 | 16.64s | Cấu hình riêng | NOT TESTED |
| 18 | trainer-recurrent-barrier.ipynb | barrier_digital | TF | 500 | 250 | Chưa đo | 2 | 13.32s | Cấu hình riêng | NOT TESTED |
| 19 | trainer-recurrent-fwdstart.ipynb | fwd_none | TF | 500 | 250 | Chưa đo | 2 | 10.82s | Cấu hình riêng | NOT TESTED |
| 20 | trainer-recurrent-fwdstart.ipynb | fwd_all_modes | TF | 500 | 250 | Chưa đo | 2 | 8.52s | Cấu hình riêng | NOT TESTED |
| 21 | trainer-recurrent-fwdstart.ipynb | fwd_aggr_2 | TF | 500 | 250 | Chưa đo | 2 | 8.16s | Cấu hình riêng | NOT TESTED |
| 22 | trainer-recurrent-fwdstart.ipynb | fwd_aggr_5bnd | TF | 500 | 250 | Chưa đo | 2 | 8.09s | Cấu hình riêng | NOT TESTED |
| 23 | trainer-recurrent-fwdstart.ipynb | fwd_aggr_5 | TF | 500 | 250 | Chưa đo | 2 | 8.87s | Cấu hình riêng | NOT TESTED |
| 24 | trainer-recurrent-fwdstart.ipynb | fwd_past_repr | TF | 500 | 250 | Chưa đo | 2 | 8.09s | Cấu hình riêng | NOT TESTED |

---

## 6. DANH MỤC LOGS CHI TIẾT
Tất cả các file log định dạng UTF-8 được lưu tại:
- `results/windows_verification_20261106_165611/logs/test_parity_jax.log`
- `results/windows_verification_20261106_165611/logs/test_parity_tf.log`
- `results/windows_verification_20261106_165611/logs/test_warmup_resume_jax.log`
- `results/windows_verification_20261106_165611/logs/test_warmup_resume_tf.log`
