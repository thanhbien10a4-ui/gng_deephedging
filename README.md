# Deep Hedging Rewritten (7 Notebooks & 24 Cấu Hình)

Bộ mã nguồn viết lại độc lập hoàn toàn cho bài toán **Deep Hedging**, chuẩn hóa cấu trúc 7 Jupyter Notebook và 24 cấu hình benchmark theo các backend đã chỉ định.

Sản phẩm được chuẩn hóa trực tiếp với hai không gian tên độc lập:
- `dh_jax`: Backend JAX thuần chức năng (Pure Functional JAX engine với `jax.jit`, `jax.grad`, `jax.lax.scan`; quản lý tham số qua `params dict`, thuật toán Adam thuần chức năng; không sử dụng Flax hoặc Optax).
- `dh_tf`: Backend TensorFlow 2 / Keras engine (chuẩn hóa trên TensorFlow 2.15.1 / Keras 2.15.0).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
deephedging-rewritten/
├── dh_jax/                                 # Module tính toán phòng hộ cho backend JAX
│   ├── world.py                            # Mô phỏng đường đi giá (BS, Stoch Vol)
│   ├── model.py                            # Mạng nơ-ron Deep Hedging, rollout và hàm loss
│   ├── train.py                            # Vòng lặp huấn luyện, optimizer Adam, clipping
│   ├── stats.py                            # Đo lường rủi ro (CVaR95, VaR, phân rã quantile)
│   ├── fd.py                               # Bộ định giá sai phân hữu hạn (Finite Difference)
│   └── plotting.py                         # Đồ họa hóa quá trình huấn luyện và delta
├── dh_tf/                                  # Module tính toán phòng hộ cho backend TensorFlow
│   ├── world.py                            # Mô phỏng thị trường và nhân quả hóa rào cản
│   ├── gym.py                              # Lớp mô hình Keras Deep Hedging Gym
│   ├── layers.py                           # Các tầng bộ nhớ hồi quy (classic, aggregate, past, event)
│   ├── objectives.py                       # Các hàm mục tiêu tiền tệ (CVaR, Exp, Exp2)
│   ├── trainer.py                          # Vòng lặp huấn luyện TF, optimizer Adam, lưu mô hình
│   └── plot_bs_hedge.py                    # Vẽ đối chiếu delta phòng hộ
├── configs/
│   ├── __init__.py
│   └── catalog.py                          # Định nghĩa chi tiết 24 cấu hình chuẩn hóa
├── tests/
│   ├── test_smoke_24.py                    # Script kiểm thử smoke 24 cấu hình
│   ├── test_parity_jax.py                  # Đối chiếu số học JAX với mã gốc (1 cấu hình đại diện)
│   ├── test_parity_tf.py                   # Đối chiếu số học TF với mã gốc (1 cấu hình đại diện)
│   ├── test_warmup_resume_jax.py           # Kiểm tra bất biến Warmup JAX
│   ├── test_warmup_resume_tf.py            # Kiểm tra bất biến Warmup TF
│   ├── test_resume_tf_exact.py             # Kiểm tra khôi phục Adam & weights TF (1->2 epoch)
│   ├── test_resume_tf_multi_epoch.py       # Kiểm tra khôi phục đa chặng 5 epoch TF (2->5, 2->3->5)
│   └── run_slurm_smoke.sbatch              # Kịch bản Slurm chạy trên compute node
├── environments/
│   ├── environment_jax.txt                 # Ghi nhận môi trường Python 3.13 & package JAX
│   └── environment_tf.txt                  # Ghi nhận môi trường Python 3.10 & package TensorFlow
├── results/                                # Kết quả, log, CSV/JSON và notebook đã thực thi
├── trainer.ipynb                           # Notebook 1: JAX (2 cấu hình: trainer_bs, trainer_sv)
├── blackscholes.ipynb                      # Notebook 2: JAX (5 cấu hình BS variants)
├── trainer-american_digital.ipynb          # Notebook 3: JAX (5 cấu hình American Digital)
├── trainer-recurrent-barrier_smalldt.ipynb # Notebook 4: JAX (2 cấu hình Barrier Put nhỏ dt)
├── trainer-bs.ipynb                        # Notebook 5: TensorFlow (2 cấu hình: coarse, fine)
├── trainer-recurrent-barrier.ipynb         # Notebook 6: TensorFlow (2 cấu hình Barrier Put)
├── trainer-recurrent-fwdstart.ipynb        # Notebook 7: TensorFlow (6 cấu hình Forward Start Put)
├── BASELINE_STATUS.md                      # Báo cáo chốt bản nền kỹ thuật toàn diện
├── KNOWN_ISSUES.md                         # Danh sách xử lý lỗi, bản vá resume & lưu ý số học
├── MAPPING_24_CONFIGS.md                   # Bảng ánh xạ 24 cấu hình chi tiết
├── TEST_REPORT.md                          # Báo cáo kiểm thử số học ban đầu
├── requirements.txt                        # Danh sách phụ thuộc môi trường
└── README.md                               # Hướng dẫn cài đặt & vận hành
```

---

## 2. Phân Bổ Backend Cho 7 Notebook (24 Cấu Hình)

| Tên Notebook | Backend Chỉ Định | Số Cấu Hình | Danh Sách Cấu Hình (IDs) | Số Epoch Gốc |
| :--- | :---: | :---: | :--- | :---: |
| `blackscholes.ipynb` | **JAX** | 5 | `bs_base`, `bs_exp2`, `bs_nodrift`, `bs_nodrift_exp2`, `bs_small` | 10 - 600 |
| `trainer-bs.ipynb` | **TensorFlow** | 2 | `trainer_bs_coarse`, `trainer_bs_fine` | 20,000 |
| `trainer.ipynb` | **JAX** | 2 | `trainer_bs`, `trainer_sv` | 800 |
| `trainer-american_digital.ipynb` | **JAX** | 5 | `am_dig_none`, `am_dig_digital`, `am_dig_standard`, `am_dig_mix_relu`, `am_dig_mix_softplus` | 500 |
| `trainer-recurrent-barrier.ipynb` | **TensorFlow** | 2 | `barrier_none`, `barrier_digital` | 250 |
| `trainer-recurrent-barrier_smalldt.ipynb` | **JAX** | 2 | `barrier_sdt_none`, `barrier_sdt_digital` | 250 |
| `trainer-recurrent-fwdstart.ipynb` | **TensorFlow** | 6 | `fwd_none`, `fwd_all_modes`, `fwd_aggr_2`, `fwd_aggr_5bnd`, `fwd_aggr_5`, `fwd_past_repr` | 2,000 |
| **TỔNG CỘNG** | **7 Notebook** | **24** | **14 JAX + 10 TensorFlow** | — |

---

## 3. Hướng Dẫn Thiết Lập Hai Môi Trường Riêng Biệt (JAX & TensorFlow)

Do JAX và TensorFlow có thể xung đột phiên bản phụ thuộc (đặc biệt trên Windows), dự án được thiết kế để chạy trên **2 môi trường tách biệt hoàn toàn**:

### Môi trường 1: JAX Environment (Khuyên dùng Python 3.13 hoặc 3.11/3.12 64-bit)
Môi trường này dùng để chạy 4 notebook JAX (`trainer.ipynb`, `blackscholes.ipynb`, `trainer-american_digital.ipynb`, `trainer-recurrent-barrier_smalldt.ipynb`).

```powershell
# 1. Tạo môi trường ảo cho JAX
python -m venv .venv-jax

# 2. Kích hoạt môi trường
.\.venv-jax\Scripts\Activate.ps1

# 3. Cài đặt các gói JAX và thư viện tính toán khoa học
pip install --upgrade pip
pip install jax jaxlib numpy scipy matplotlib ipykernel nbclient nbformat

# 4. Đăng ký kernel Jupyter riêng cho JAX
python -m ipykernel install --user --name dh-rewritten-jax --display-name "Python (dh-rewritten-jax)"
```

### Môi trường 2: TensorFlow Environment (Khuyên dùng Python 3.10 64-bit)
Môi trường này dùng để chạy 3 notebook TensorFlow (`trainer-bs.ipynb`, `trainer-recurrent-barrier.ipynb`, `trainer-recurrent-fwdstart.ipynb`).

```powershell
# 1. Tạo môi trường ảo cho TensorFlow (sử dụng Python 3.10)
py -3.10 -m venv .venv-tf

# 2. Kích hoạt môi trường
.\.venv-tf\Scripts\Activate.ps1

# 3. Cài đặt đúng phiên bản TensorFlow 2.15.1 và Keras tương thích
pip install --upgrade pip
pip install tensorflow==2.15.1 keras==2.15.0 tensorflow-probability==0.23.0 cdxbasics==0.2.141 numpy==1.26.4 scipy==1.15.2 matplotlib ipykernel

# 4. Đăng ký kernel Jupyter riêng cho TensorFlow
python -m ipykernel install --user --name dh-rewritten-tf --display-name "Python (dh-rewritten-tf)"
```

---

## 4. Khởi Chạy & Kiểm Thử

### Chế độ Smoke Test (Nhanh - 2 epoch):
Thiết lập biến môi trường `DH_MODE="smoke"` để chạy kiểm thử luồng thực thi nhanh (2 epoch, tối đa 500 mẫu):

```powershell
# Trên Windows PowerShell
$env:DH_MODE = "smoke"

# Chạy smoke test tự động toàn bộ 24 cấu hình qua runner
python run_windows_smoke.py

# Hoặc chạy riêng 3 notebook TensorFlow sau bản vá
python run_windows_tf_smoke.py
```

### Chế độ Huấn luyện gốc (Full Hyperparameters):
Để huấn luyện với đầy đủ tham số và số epoch gốc của tác giả:

```powershell
$env:DH_MODE = "original"
jupyter notebook
```
Khi mở notebook trong Jupyter Notebook hoặc VS Code, chọn đúng kernel đã đăng ký:
- Các notebook JAX: Chọn kernel **`Python (dh-rewritten-jax)`**.
- Các notebook TensorFlow: Chọn kernel **`Python (dh-rewritten-tf)`**.

---

## 5. Tóm Tắt Kết Quả Kiểm Thử & Giới Hạn Hỗ Trợ

1. **Kiểm thử Smoke Test 24/24 cấu hình:** Đạt **24/24 PASS** trên Windows CPU (14 cấu hình JAX và 10 cấu hình TensorFlow hoàn tất 2 epoch, loss và trọng số hữu hạn).
2. **Kiểm thử đối chiếu số học Parity:** Đạt sai số tuyệt đối `0.00e+00` (dưới ngưỡng $\le 10^{-5}$) trên **1 cấu hình đại diện mỗi backend** (`trainer_bs` cho JAX và `trainer_bs_coarse` cho TensorFlow). *Lưu ý:* Kết quả này xác nhận tính tương đương số học trên 2 cấu hình đại diện, không suy rộng trạng thái PASS parity cho toàn bộ 24 cấu hình do các cấu hình khác có hàm loss và cấu trúc đặc trưng riêng.
3. **Bất biến Warmup:** Đạt độ lệch tham số `0.00` trước bước cập nhật đầu tiên trên cả JAX và TensorFlow.
4. **Bản vá Checkpoint Resume TensorFlow:**
   - Đã sửa lỗi return sớm; khôi phục hoàn chỉnh trọng số mô hình `last_weights` và các slot biến Adam ($m$, $v$, `iterations`).
   - Đã kiểm chứng qua kịch bản đa chặng 5 epoch (liên tục vs 2->5 vs 2->3->5): sai lệch trọng số và hàm loss đạt `0.00e+00`, trật tự batch theo chỉ số mẫu khớp 100%.
   - **Giới hạn hỗ trợ:** Cơ chế hiện tại **chỉ hỗ trợ khôi phục tại ranh giới kết thúc epoch** (`on_epoch_end`), **không hỗ trợ resume giữa chừng trong một epoch (mid-epoch / intra-batch)**.
   - **Phụ thuộc môi trường:** Dựa trên `keras.src.engine.data_adapter.DataHandler` của Keras 2.15. Checkpoint thiếu trạng thái optimizer sẽ chủ động báo `RuntimeError`, từ chối chạy lại âm thầm từ đầu.
5. **Cảnh báo phương pháp luận:** Smoke test thành công (2 epoch) chỉ kiểm tra tính toàn vẹn kỹ thuật (numerical execution integrity), **không đồng nghĩa với việc mô hình đã hội tụ tối ưu hoặc đạt chất lượng phòng hộ tài chính**. Để quan sát động học huấn luyện, cần chạy đủ số mẫu và số epoch gốc.
