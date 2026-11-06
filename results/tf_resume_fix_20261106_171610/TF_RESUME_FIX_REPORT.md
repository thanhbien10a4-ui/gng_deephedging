# BÁO CÁO SỬA LỖI RESUME CHECKPOINT TENSORFLOW (`dh_tf`)

**Thư mục lưu trữ**: `results/tf_resume_fix_20261106_171610/`  
**Thời gian thực hiện**: 2026-11-06 17:16 – 17:50 (UTC+7)  
**Môi trường thực thi**: Windows 11 CPU · TensorFlow 2.15.1 (`D:\deephedging-rewritten\.venv-tf\Scripts\python.exe`)  
**Tham chiếu**: `results/windows_verification_20261106_165611/VERIFICATION_REPORT.md` (mục TF Resume = FAIL với sai lệch `2.04e-03`; báo cáo và các thư mục kết quả cũ được giữ nguyên 100%).

> [!IMPORTANT]
> - Đây là bản sửa **cơ chế lưu/nạp checkpoint và khôi phục vòng lặp huấn luyện** của TensorFlow so với bản bàn giao (lỗi `return True` sớm trong `restore_from_cache()` vốn cũng tồn tại trong mã tham chiếu `D:\deephedging\gym.py`; thư mục mã tham chiếu được giữ nguyên không sửa).
> - Bản sửa này **không** đồng nghĩa với việc đã kiểm thử đủ 24 cấu hình: kiểm chứng resume mới được thực hiện trên 2 cấu hình đại diện (1 feedforward, 1 recurrent, cả trường hợp 1 batch/epoch và nhiều batch/epoch) cùng test hồi quy parity TensorFlow hiện có.

---

## 1. NGUYÊN NHÂN GỐC RỄ (KIỂM TRA TRỰC TIẾP TRONG `dh_tf/gym.py` & `dh_tf/trainer.py`)

1. **`return True` sớm trong `VanillaDeepHedgingGym.restore_from_cache()` (`dh_tf/gym.py` dòng 371)**:
   - Ngay sau khi gán `self.set_weights(gym_weights)`, hàm trả về `True` lập tức, biến toàn bộ đoạn nạp trạng thái optimizer phía dưới thành mã chết (unreachable code). Khi nạp lại trong tiến trình mới, Adam khởi tạo mới hoàn toàn ($m = 0, v = 0, \text{iterations} = 0$).
2. **Slot biến của Keras Optimizer (`m`, `v`, `iterations`) được khởi tạo lười (lazy build)**:
   - Trong TensorFlow / Keras $\ge 2.11$, sau `gym.compile(optimizer=...)`, `self.optimizer.variables` vẫn rỗng cho tới bước cập nhật gradient đầu tiên. Do đó trước khi khôi phục, phải gọi `self.optimizer.build(self.trainable_variables)` và kiểm tra số lượng + kích thước (`shape`) từng slot khớp với checkpoint.
3. **Lỗi tính `self_opt_uid` trong `restore_from_cache()`**:
   - Mã cũ tính `self_id_config` từ `opt_config` (của cache) thay vì `self_opt_config` (của optimizer hiện tại), làm phép đối chiếu `opt_uid != self_opt_uid` vô tác dụng.
4. **Bảo toàn `last_weights` đi kèm trạng thái Adam tại đúng bước đó vs `best_weights`**:
   - Trong `Monitor.finalize()` (`dh_tf/trainer.py`), `self.write_cache()` được gọi **trước** `self.progress_data.set_best_weights(...)`. Như vậy `cache['gym']['gym_weights']` luôn lưu `last_weights` tại thời điểm kết thúc epoch cuối cùng (khớp đúng với `opt_weights` của bước đó), còn `best_weights` được lưu tách biệt trong `cache['progress_data'].best_weights` chỉ để phục vụ đánh giá sau khi kết thúc huấn luyện. Luồng tách biệt này được giữ nguyên.
5. **Bảo toàn epoch đã hoàn tất và cơ chế sinh thứ tự batch của Keras khi `initial_epoch > 0`**:
   - Mã cũ trong `dh_tf/trainer.py` gọi `gym.fit(..., epochs = epochs - (monitor.current_epoch + 1))` mà không truyền `initial_epoch = monitor.current_epoch + 1`.
   - Trong Keras `TensorLikeDataAdapter`, cơ chế sinh thứ tự batch dùng `tf.random.shuffle(indices)` lặp qua các epoch trên một `tf.data.Iterator`. Khi resume ở tiến trình mới với `initial_epoch > 0`, nếu `DataHandler` tạo iterator mới từ đầu mà không tua qua `initial_epoch * inferred_steps` batch của các epoch đã hoàn tất, epoch tiếp theo sẽ lấy lại thứ tự batch của epoch 0 thay vì epoch `initial_epoch`.

---

## 2. CÁC THAY ĐỔI ĐÃ THỰC HIỆN

1. **[`dh_tf/gym.py`](file:///D:/deephedging-rewritten/dh_tf/gym.py)** (bản gốc trước sửa: [`before/gym.py`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/before/gym.py), diff: [`gym.py.diff`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/gym.py.diff)):
   - `create_cache()`: Lưu `gym_weights` (`last_weights`) và `opt_weights` dưới dạng mảng NumPy theo đúng thứ tự `self.optimizer.variables` (`[iterations, m_0..m_k, v_0..v_k]`).
   - `restore_from_cache()`:
     - Kiểm tra `gym_uid`, sự tồn tại của `self.optimizer`, và `self_opt_uid` tính từ `self_opt_config`.
     - Nếu checkpoint thiếu `opt_weights` (rỗng hoặc `None`), **cảnh báo rõ không hỗ trợ resume chính xác và trả về `False`**, tuyệt đối không gán `gym_weights` và không âm thầm chạy tiếp với Adam mới.
     - Gọi `self.optimizer.build(self.trainable_variables)` nếu các slot chưa được tạo; kiểm tra số lượng biến và `shape` của từng biến khớp 100% trước khi gán.
     - Nạp `self.set_weights(gym_weights)` (trọng số `last_weights`) và `v.assign(w)` cho toàn bộ biến của optimizer (`iterations`, `m`, `v`), sau đó đọc lại xác minh `np.array_equal(v.numpy(), w)` cho mọi biến.
   - `fit()`: Khi `initial_epoch > 0`, giữ nguyên hoàn toàn cơ chế `TensorLikeDataAdapter` (`tf.random.shuffle`) của Keras cho tổng số `epochs` và dùng `dh._dataset = dh._dataset.skip(initial_epoch * dh.inferred_steps)` để tua qua đúng số batch của các epoch đã hoàn tất trước đó.
2. **[`dh_tf/trainer.py`](file:///D:/deephedging-rewritten/dh_tf/trainer.py)** (bản gốc trước sửa: [`before/trainer.py`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/before/trainer.py), diff: [`trainer.py.diff`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/trainer.py.diff)):
   - Truyền `epochs = epochs, initial_epoch = monitor.current_epoch + 1` vào `gym.fit(...)`.
   - Cập nhật thông báo khi `restore_from_cache()` trả về `False` (không nạp `progress_data`, không báo resume thành công).
3. **Không thay đổi**: công thức mô hình, payoff, hàm loss, learning rate, clipping, hay cơ chế sinh hoán vị batch của Keras.

---

## 3. KẾT QUẢ KIỂM CHỨNG

Quy trình kiểm thử trong [`tests/test_resume_tf_exact.py`](file:///D:/deephedging-rewritten/tests/test_resume_tf_exact.py) và [`tests/test_warmup_resume_tf.py`](file:///D:/deephedging-rewritten/tests/test_warmup_resume_tf.py):
- **Tiến trình A**: Chạy liên tục 2 epoch từ đầu.
- **Tiến trình B**: Chạy 1 epoch, lưu checkpoint ra đĩa.
- **Tiến trình C (tiến trình Python mới)**: Nạp checkpoint từ Tiến trình B, chạy tiếp epoch 2.
- Đối chiếu trực tiếp giữa A và C trên: `last_weights`, các slot Adam `m` và `v`, `iterations`, `train_loss`, `val_loss`, và `best_epoch` (giữ nguyên ngưỡng sai số $\le 10^{-5}$, `iterations` khớp tuyệt đối).

### 3.1. Bảng kết quả đối chiếu Resume (Tiến trình mới vs Chạy liên tục 2 epoch)

| Cấu hình kiểm thử | Kiến trúc | Batch size (số batch/epoch) | Iterations (Ep1 $\to$ Ep2) | Max Abs Diff `last_weights` | Max Abs Diff Adam `m` | Max Abs Diff Adam `v` | Abs Diff `train_loss` | Abs Diff `val_loss` | Ngưỡng | Kết quả |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Feedforward (`ff`)** | `depth=2, width=10, softplus` | 32 (4 batch/ep) | `4 -> 8` (khớp `8 == 8`) | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | $\le 10^{-5}$ | **PASS** |
| **Feedforward (`ff`)** | `depth=2, width=10, softplus` | 100 (1 batch/ep) | `1 -> 2` (khớp `2 == 2`) | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | $\le 10^{-5}$ | **PASS** |
| **Recurrent (`rec`)** | `classic=1, depth=2, width=10` | 32 (4 batch/ep) | `4 -> 8` (khớp `8 == 8`) | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | $\le 10^{-5}$ | **PASS** |
| **Recurrent (`rec`)** | `classic=1, depth=2, width=10` | 100 (1 batch/ep) | `1 -> 2` (khớp `2 == 2`) | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | $\le 10^{-5}$ | **PASS** |
| **`test_warmup_resume_tf.py`** | Feedforward (`batch_size=32`) | 32 (4 batch/ep) | `4 -> 8` | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | **`0.00e+00`** | $\le 10^{-5}$ | **PASS** |

### 3.2. Kiểm tra từ chối Checkpoint thiếu trạng thái Optimizer & Hồi quy Parity TensorFlow

| Bài kiểm tra | Bằng chứng thực nghiệm | Kết quả |
| :--- | :--- | :---: |
| **Checkpoint cũ thiếu `opt_weights`** | `restore_from_cache()` cảnh báo rõ không hỗ trợ resume chính xác, trả về `False`, giữ nguyên 100% trọng số hiện tại (`weights_untouched = True`), không âm thầm khởi tạo Adam mới rồi báo resume thành công. | **PASS** |
| **Hồi quy Parity TF (`tests/test_parity_tf.py`)** | Spot diff: `0.00e+00`, Rollout Actions diff: `0.00e+00`, Initial Loss RelDiff: `0.00e+00`, Gradients diff: `0.00e+00`, 1 Adam Step Param diff: `0.00e+00` (Ngưỡng $\le 10^{-5}$). | **PASS** |

---

## 4. DANH MỤC TỆP TRONG `results/tf_resume_fix_20261106_171610/`
- Bản sao trước khi sửa:
  - [`before/gym.py`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/before/gym.py)
  - [`before/trainer.py`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/before/trainer.py)
- Bản diff chi tiết:
  - [`gym.py.diff`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/gym.py.diff)
  - [`trainer.py.diff`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/trainer.py.diff)
- Kết quả JSON & Logs kiểm thử:
  - [`tf_resume_results.json`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/tf_resume_results.json)
  - [`logs/test_resume_tf_exact.log`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/logs/test_resume_tf_exact.log)
  - [`logs/tf_resume_phases.log`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/logs/tf_resume_phases.log)
  - [`logs/test_warmup_resume_tf.log`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/logs/test_warmup_resume_tf.log)
  - [`logs/test_parity_tf.log`](file:///D:/deephedging-rewritten/results/tf_resume_fix_20261106_171610/logs/test_parity_tf.log)
