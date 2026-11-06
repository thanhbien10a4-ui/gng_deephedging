# Báo Cáo Chốt Bản Nền Kỹ Thuật (Baseline Technical Status) - Revision 1

**Dự án**: Deep Hedging Rewritten (7 Notebooks & 24 Cấu Hình)  
**Thời điểm bàn giao (Timestamp)**: `20261106_185500_rev1`  
**Môi trường phần cứng**: Windows 11 CPU (64-bit)  
**Môi trường thực thi JAX**: Python 3.13.4 (`C:\Users\Hi Windows 11 Home\AppData\Local\Programs\Python\Python313\python.exe`)  
**Môi trường thực thi TensorFlow**: Python 3.10.11 / TensorFlow 2.15.1 (`D:\deephedging-rewritten\.venv-tf\Scripts\python.exe`)  

---

## 1. Xác Minh Môi Trường JAX & Kiến Trúc Mã Nguồn Thực Tế

### A. Đối soát phiên bản JAX & jaxlib
- **Kiểm tra trực tiếp từ interpreter Python 3.13 (`sys.executable`)**:
  - `sys.executable`: `C:\Users\Hi Windows 11 Home\AppData\Local\Programs\Python\Python313\python.exe`
  - `jax.__version__`: **`0.12.0.1`**
  - `jaxlib.__version__`: **`0.12.0.1`**
  - `jax.__file__`: `C:\Users\Hi Windows 11 Home\AppData\Local\Programs\Python\Python313\Lib\site-packages\jax\__init__.py`
- **Đối chiếu lịch sử tệp môi trường**:
  - Tệp lưu trữ lúc chạy smoke (`results/windows_smoke_20261106_163915/environment/packages_jax.txt`, dòng 71-72): `jax 0.12.0.1`, `jaxlib 0.12.0.1`.
  - Tệp lưu trữ lúc bàn giao (`environments/environment_jax.txt`, dòng 76-77): `jax 0.12.0.1`, `jaxlib 0.12.0.1`.
- **Đính chính thông tin**:
  - Môi trường JAX xuyên suốt từ lúc bắt đầu kiểm thử đến hiện tại là **JAX 0.12.0.1 / jaxlib 0.12.0.1** (bản dựng wheel tương thích Windows trên Python 3.13).
  - Không có bất kỳ thay đổi phiên bản hay cài đặt lại package nào. Con số `0.4.38` xuất hiện trong đoạn văn thuyết minh ở lượt trả lời trước là sự nhầm lẫn phát sinh trong văn bản trả lời, hoàn toàn không phản ánh môi trường thực tế trên máy.

### B. Kiến trúc mã nguồn `dh_jax` (Xác minh Flax / Optax)
- **Kiểm tra import thực tế**: Rà soát toàn bộ các tệp nguồn trong `dh_jax/` (`fd.py`, `model.py`, `plotting.py`, `stats.py`, `train.py`, `world.py`).
- **Kết luận**:
  - Module `dh_jax` là **Pure Functional JAX engine**: sử dụng trực tiếp các hàm nguyên bản của JAX (`jax.jit`, `jax.grad`, `jax.value_and_grad`, `jax.lax.scan`), mảng tính toán `jax.numpy`, hàm kích hoạt `jax.nn`.
  - Mô hình mạng nơ-ron quản lý trọng số dạng từ điển tham số thuần (`params dict`), khởi tạo qua hàm `init_params()`.
  - Vòng lặp tối ưu sử dụng thuật toán Adam thuần chức năng tự viết trong `dh_jax/train.py` (`AdamState(step, m, v)`).
  - Mã nguồn **hoàn toàn không import và không sử dụng Flax (`flax.linen`) hoặc Optax**, dù các gói này có thể được cài trong môi trường. Mô tả chuẩn xác là **Pure Functional JAX**, không phải "JAX / Flax".

---

## 2. Ánh Xạ Chuẩn Hóa 7 Notebook & 24 Cấu Hình Sang Backend

Toàn bộ 24 cấu hình benchmark được chuẩn hóa độc lập vào 2 không gian tên backend (`dh_jax` và `dh_tf`). Số epoch và số mẫu trong chế độ gốc (`original`) khác biệt rõ rệt theo thiết kế từng bài toán:

| STT | Tên Notebook | Backend | Cấu Hình ID | Loại Sản Phẩm Phái Sinh | Số Bước ($nSteps$) | $dt$ | Số Mẫu Huấn Luyện (Smoke / Gốc) | Số Epoch Huấn Luyện (Smoke / Gốc) | Hàm Mục Tiêu / Loss |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `blackscholes.ipynb` | **JAX** | `bs_base` | Black-Scholes Call Benchmark | 20 | 0.02 | 500 / 50,000 | 2 / **400** | CVaR 95% |
| 2 | `blackscholes.ipynb` | **JAX** | `bs_exp2` | BS Call Benchmark (Exponential loss) | 20 | 0.02 | 500 / 50,000 | 2 / **400** | Exponential loss |
| 3 | `blackscholes.ipynb` | **JAX** | `bs_nodrift` | BS Zero Drift Benchmark | 20 | 0.02 | 500 / 50,000 | 2 / **600** | CVaR 95% |
| 4 | `blackscholes.ipynb` | **JAX** | `bs_nodrift_exp2` | BS Zero Drift (Exponential Loss) | 20 | 0.02 | 500 / 50,000 | 2 / **600** | Exponential loss |
| 5 | `blackscholes.ipynb` | **JAX** | `bs_small` | BS Tham Số Nhỏ (Kiểm thử nhanh) | 20 | 0.02 | 500 / 10,000 | 2 / **10** | CVaR 95% |
| 6 | `trainer-bs.ipynb` | **TensorFlow** | `trainer_bs_coarse` | BS Vanilla Hedging (Bước thô) | 20 | 0.02 | 500 / 10,000 | 2 / **20,000** | Exp Utility loss |
| 7 | `trainer-bs.ipynb` | **TensorFlow** | `trainer_bs_fine` | BS Vanilla Hedging (Bước mịn) | 100 | 0.004 | 500 / 10,000 | 2 / **20,000** | Exp Utility loss |
| 8 | `trainer.ipynb` | **JAX** | `trainer_bs` | BS Vanilla Call/Put | 20 | 0.02 | 500 / 10,000 | 2 / **800** | CVaR 95% |
| 9 | `trainer.ipynb` | **JAX** | `trainer_sv` | Heston Stochastic Volatility Call | 20 | 0.02 | 500 / 10,000 | 2 / **800** | CVaR 95% |
| 10 | `trainer-american_digital.ipynb` | **JAX** | `am_dig_none` | American Digital Option (Baseline) | 100 | 0.02 | 500 / 20,000 | 2 / **500** | Exponential loss |
| 11 | `trainer-american_digital.ipynb` | **JAX** | `am_dig_digital` | American Digital Option (Digital indicator) | 100 | 0.02 | 500 / 20,000 | 2 / **500** | Exponential loss |
| 12 | `trainer-american_digital.ipynb` | **JAX** | `am_dig_standard` | American Digital Option (Standard features)| 100 | 0.02 | 500 / 20,000 | 2 / **500** | Exponential loss |
| 13 | `trainer-american_digital.ipynb` | **JAX** | `am_dig_mix_relu` | American Digital (Hỗn hợp ReLU) | 100 | 0.02 | 500 / 20,000 | 2 / **500** | Exponential loss |
| 14 | `trainer-american_digital.ipynb` | **JAX** | `am_dig_mix_softplus`| American Digital (Hỗn hợp Softplus) | 100 | 0.02 | 500 / 20,000 | 2 / **500** | Exponential loss |
| 15 | `trainer-recurrent-barrier.ipynb` | **TensorFlow** | `barrier_none` | Barrier Put Hedging (Không đặc trưng) | 100 | 0.004 | 500 / 20,000 | 2 / **250** | Exp Utility loss |
| 16 | `trainer-recurrent-barrier.ipynb` | **TensorFlow** | `barrier_digital` | Barrier Put Hedging (Digital indicator) | 100 | 0.004 | 500 / 20,000 | 2 / **250** | Exp Utility loss |
| 17 | `trainer-recurrent-barrier_smalldt.ipynb` | **JAX** | `barrier_sdt_none` | Barrier Option $dt$ nhỏ (Không đặc trưng) | 20 | 0.02 | 500 / 20,000 | 2 / **250** | CVaR 95% |
| 18 | `trainer-recurrent-barrier_smalldt.ipynb` | **JAX** | `barrier_sdt_digital` | Barrier Option $dt$ nhỏ (Digital indicator) | 20 | 0.02 | 500 / 20,000 | 2 / **250** | CVaR 95% |
| 19 | `trainer-recurrent-fwdstart.ipynb` | **TensorFlow** | `fwd_none` | Forward Start Put (Không đặc trưng) | 20 | 0.004 | 500 / 20,000 | 2 / **2,000** | Exp Utility loss |
| 20 | `trainer-recurrent-fwdstart.ipynb` | **TensorFlow** | `fwd_all_modes` | Forward Start Put (Đầy đủ mode) | 20 | 0.004 | 500 / 20,000 | 2 / **2,000** | Exp Utility loss |
| 21 | `trainer-recurrent-fwdstart.ipynb` | **TensorFlow** | `fwd_aggr_2` | Forward Start Put (Tổng hợp 2 trạng thái) | 20 | 0.004 | 500 / 20,000 | 2 / **2,000** | Exp Utility loss |
| 22 | `trainer-recurrent-fwdstart.ipynb` | **TensorFlow** | `fwd_aggr_5bnd` | Forward Start Put (5 trạng thái có chặn) | 20 | 0.004 | 500 / 20,000 | 2 / **2,000** | Exp Utility loss |
| 23 | `trainer-recurrent-fwdstart.ipynb` | **TensorFlow** | `fwd_aggr_5` | Forward Start Put (5 trạng thái không chặn) | 20 | 0.004 | 500 / 20,000 | 2 / **2,000** | Exp Utility loss |
| 24 | `trainer-recurrent-fwdstart.ipynb` | **TensorFlow** | `fwd_past_repr` | Forward Start Put (Biểu diễn quá khứ) | 20 | 0.004 | 500 / 20,000 | 2 / **2,000** | Exp Utility loss |

---

## 3. Các Kiểm Thử Kỹ Thuật Đã Thực Hiện & Phạm Vi Thực Tế

| Nội Dung Kiểm Thử | Công Cụ / Kịch Bản | Interpreter | Phạm Vi Kiểm Chứng Thực Tế | Tiêu Chí Đánh Giá | Kết Quả Thực Tế | Bằng Chứng Lưu Trữ |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Smoke Test Toàn Bộ 24 Cấu Hình** | `run_windows_smoke.py` | Python 3.13 (JAX) & .venv-tf (TF) | 7 notebook, 24 cấu hình (14 JAX + 10 TF) | Hoàn tất đúng 2 epoch, 500 mẫu, loss hữu hạn, trọng số hữu hạn | **PASS (24/24)** | `results/windows_smoke_20261106_163915/` |
| **Đối Chiếu Số Học Parity JAX** | `tests/test_parity_jax.py` | Python 3.13 | **1 cấu hình đại diện** (`trainer_bs`, 1/14 JAX) | Sai số rollout, loss, grad, Adam step <= 10^-5 | **PASS (`0.00e+00`)** | `results/windows_verification_20261106_165611/` |
| **Đối Chiếu Số Học Parity TensorFlow** | `tests/test_parity_tf.py` | `.venv-tf` Python 3.10 | **1 cấu hình đại diện** (`trainer_bs_coarse`, 1/10 TF) | Sai số rollout, loss, grad, Adam step <= 10^-5 | **PASS (`0.00e+00`)** | `results/windows_verification_20261106_165611/` |
| **Kiểm Tra Bất Biến Warmup (JAX)** | `tests/test_warmup_resume_jax.py` | Python 3.13 | JAX model warmup | Trọng số & tham số trước/sau warmup khớp 100% | **PASS (`0.00e+00`)** | `results/windows_verification_20261106_165611/` |
| **Kiểm Tra Bất Biến Warmup (TF)** | `tests/test_warmup_resume_tf.py` | `.venv-tf` Python 3.10 | TF model warmup | Trọng số & tham số trước/sau warmup khớp 100% | **PASS (`0.00e+00`)** | `results/windows_verification_20261106_165611/` |
| **Kiểm Tra & Sửa Lỗi Resume TF (1 -> 2 Epoch)** | `tests/test_resume_tf_exact.py` | `.venv-tf` Python 3.10 | 2 cấu hình TF: feedforward (`ff`) & recurrent (`rec`) | Sai số `last_weights`, Adam m, v, train/val loss <= 10^-5 | **PASS (`0.00e+00`)** | `results/tf_resume_fix_20261106_171610/` |
| **Kiểm Tra Resume Đa Epoch (5 Epochs, 4 Batches/Epoch)** | `tests/test_resume_tf_multi_epoch.py` | `.venv-tf` Python 3.10 | 2 cấu hình đại diện: `ff` và `rec` (Liên tục 5 ep vs 2 -> 5 vs 2 -> 3 -> 5) | Sai số <= 10^-5, iterations khớp 20, trật tự chỉ số batch khớp 100% | **PASS (`0.00e+00`)** | `results/windows_tf_verification_20261106_175449/` |
| **Đo Lường Thời Gian Khôi Phục & Tua Iterator** | `tests/test_resume_tf_multi_epoch.py` | `.venv-tf` Python 3.10 | 2 cấu hình: `ff` và `rec` | Ghi riêng `restore_time` và `skip_time` | **Đã đo**: Restore 27-35ms, Skip 2.5-4.5ms | `multi_epoch_resume_summary.json` |
| **Từ Chối Checkpoint Lỗi Thời / Thiếu Trạng Thái Adam** | `tests/test_resume_tf_multi_epoch.py` | `.venv-tf` Python 3.10 | Checkpoint giả lập bị xóa `opt_weights` | Ném `RuntimeError`, từ chối chạy lại âm thầm | **PASS** (Exit code 1, raise `RuntimeError`) | `multi_epoch_resume_summary.json` |
| **Smoke Test Riêng 3 Notebook TF Sau Sửa Lỗi** | `run_windows_tf_smoke.py` | `.venv-tf` Python 3.10 | 3 notebook TF, 10 cấu hình | 2 epoch, 500 mẫu, loss hữu hạn, trọng số hữu hạn | **PASS (10/10)** | `results/windows_smoke_tf_20261106_180940/` |

> [!NOTE]
> **Giới hạn phạm vi kiểm thử Parity**:
> - Kiểm thử đối chiếu số học bitwise (Parity test) hiện tại **mới chỉ thực hiện trên 1 cấu hình đại diện cho mỗi backend**:
>   - JAX: cấu hình `trainer_bs` (1/14 cấu hình JAX).
>   - TensorFlow: cấu hình `trainer_bs_coarse` (1/10 cấu hình TensorFlow).
> - Kết quả kiểm thử trên 2 cấu hình đại diện này đạt độ lệch tuyệt đối `0.00e+00`. Tuy nhiên, về mặt phương pháp luận, **không suy rộng trạng thái PASS parity này cho toàn bộ 24 cấu hình** do các cấu hình khác (như Barrier, Forward Start, American Digital) có cấu trúc đặc trưng và hàm loss đặc thù riêng chưa chạy parity đối đầu trực tiếp từng bước.

---

## 4. Giới Hạn Hỗ Trợ Của Checkpoint Resume TensorFlow & Phụ Thuộc Kỹ Thuật

1. **Giới hạn biên độ Epoch (Chỉ hỗ trợ tại cuối Epoch)**:
   - Checkpoint được serialize và lưu trữ tại sự kiện kết thúc epoch (`on_epoch_end`). Do đó, bản sửa checkpoint resume hiện tại **chỉ hỗ trợ khôi phục tại ranh giới kết thúc epoch** (`initial_epoch = completed_epoch + 1`).
   - Hệ thống **chưa hỗ trợ và không tuyên bố hỗ trợ khôi phục giữa chừng trong một epoch (mid-epoch / intra-batch)**.
2. **Phụ thuộc kỹ thuật vào TensorFlow / Keras 2.15**:
   - Cơ chế khôi phục chuỗi batch liên tục qua nhiều epoch dựa trên cấu trúc `keras.src.engine.data_adapter.DataHandler` và thuộc tính nội bộ `dh._dataset.skip(init_ep * steps_per_epoch)`.
   - Môi trường chuẩn hóa là TensorFlow 2.15.1 / Keras 2.15.0. Nếu chuyển sang Keras 3.x, kiến trúc DataAdapter thay đổi hoàn toàn và cần phải viết lại adapter hook này.
3. **Cơ chế an toàn từ chối Checkpoint lỗi thời**:
   - Khi file cache/checkpoint thiếu trạng thái optimizer Adam, hệ thống sẽ kích hoạt ngoại lệ `RuntimeError` thông báo chi tiết cho người dùng thay vì âm thầm khởi tạo optimizer mới để chạy tiếp.

---

## 5. Cảnh Báo Phương Pháp Luận Về Kết Quả Smoke Test

> [!WARNING]
> **Smoke Test thành công KHÔNG đồng nghĩa với việc mô hình đã hội tụ tối ưu hoặc đạt chất lượng chiến lược phòng hộ!**

- **Bản chất của Smoke Test**:
  - Mỗi cấu hình chỉ được huấn luyện đúng **2 epoch** với số mẫu rút gọn ($N = 500$).
  - Mục đích duy nhất của Smoke Test là kiểm tra **tính toàn vẹn kỹ thuật (numerical execution integrity)**: xác nhận mã nguồn không có lỗi cú pháp, tương thích shape tensor qua các chiều thời gian, đạo hàm tự động (autodiff) sinh gradient hữu hạn, optimizer cập nhật trọng số bình thường và không xảy ra hiện tượng tràn số (`NaN`, `Inf`).
- **Về việc hội tụ và chất lượng chiến lược phòng hộ**:
  - Để đánh giá động học huấn luyện và chất lượng chiến lược phòng hộ theo thiết kế gốc của tác giả, cần chạy với số mẫu gốc ($10,000 - 50,000$ mẫu) và số epoch gốc (khác biệt theo từng cấu hình: từ 10 epoch cho `bs_small`, 250 epoch cho các cấu hình barrier, 400 - 600 epoch cho blackscholes, 500 epoch cho american digital, 800 epoch cho trainer JAX, 2,000 epoch cho forward start, tới 20,000 epoch cho trainer-bs TF).
  - Ngay cả khi chạy đủ số epoch gốc, **không có sự bảo đảm mô hình chắc chắn sẽ đạt hội tụ tối ưu toàn cục** do quá trình tối ưu phi tuyến phụ thuộc vào khởi tạo ngẫu nhiên, kiến trúc mạng, learning rate schedule và tính phi lồi của bề mặt hàm mất mát trong không gian mẫu hữu hạn.

---

## 6. Danh Mục Vị Trí Các Checkpoint Kiểm Thử Cần Thiết Để Tái Lập

1. `checkpoints/`:
   - Chứa 14 tệp `.pkl` lưu trữ trọng số và tham số kiểm thử cho 14 cấu hình JAX.
   - Dùng để phục hồi nhanh và kiểm chứng số học độc lập trên môi trường JAX.
2. `results/windows_smoke_20261106_163915/isolated_checkpoints/`:
   - Chứa các checkpoint độc lập được sinh ra trong đợt chạy smoke test đầu tiên của toàn bộ 24 cấu hình.
3. `results/windows_smoke_tf_20261106_180940/isolated_checkpoints/`:
   - Chứa các checkpoint độc lập được sinh ra trong đợt tái kiểm thử smoke 10 cấu hình TensorFlow sau khi vá lỗi resume.
4. `results/tf_resume_fix_20261106_171610/before/`:
   - Chứa bản sao lưu nguyên bản của `dh_tf/gym.py` và `dh_tf/trainer.py` trước khi thực hiện bản vá.
