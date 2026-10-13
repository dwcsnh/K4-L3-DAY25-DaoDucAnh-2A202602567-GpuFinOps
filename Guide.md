# Hướng dẫn Hoàn thành Lab 25 — GPU FinOps Optimization

> Hướng dẫn từng bước dành cho sinh viên thực hiện Lab 25.
> Yêu cầu: Python 3.9+, không cần GPU, không cần cloud account, không cần API key.

---

## Mục lục

1. [Hiểu bối cảnh bài lab](#1-hiểu-bối-cảnh-bài-lab)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Khám phá dữ liệu đầu vào](#3-khám-phá-dữ-liệu-đầu-vào)
4. [Mission 1 — Kiểm toán hiệu quả GPU](#4-mission-1--kiểm-toán-hiệu-quả-gpu)
5. [Mission 2 — Đòn bẩy chi phí Inference](#5-mission-2--đòn-bẩy-chi-phí-inference)
6. [Mission 3 — Chiến lược mua GPU](#6-mission-3--chiến-lược-mua-gpu)
7. [Mission 4 — Phân bổ chi phí](#7-mission-4--phân-bổ-chi-phí)
8. [Mission 5 — Báo cáo tổng hợp](#8-mission-5--báo-cáo-tổng-hợp)
9. [Chạy toàn bộ kiểm tra](#9-chạy-toàn-bộ-kiểm-tra)
10. [Phần mở rộng "Your Turn"](#10-phần-mở-rộng-your-turn)
11. [Chuẩn bị nộp bài](#11-chuẩn-bị-nộp-bài)
12. [Các lỗi thường gặp](#12-các-lỗi-thường-gặp)

---

## 1. Hiểu bối cảnh bài lab

### Câu chuyện

Bạn là FinOps Engineer mới gia nhập **NimbusAI** — một startup LLM đang tiêu tốn quá nhiều tiền cho GPU. Nhiệm vụ của bạn là phân tích dữ liệu và **cắt giảm ít nhất 40% chi phí GPU**, đo bằng `$/1M-token`.

### Tại sao `$/1M-token` quan trọng hơn `$/GPU-giờ`?

`$/GPU-giờ` nói lên bạn **trả bao nhiêu** để thuê GPU, nhưng không nói lên bạn **nhận được gì**. Hai đội có thể trả cùng `$/GPU-giờ`, nhưng một đội phục vụ 10× nhiều token hơn do tối ưu tốt hơn. Đơn vị `$/1M-token` bắt buộc bạn đo cả **hiệu quả sử dụng**.

### Các khái niệm cốt lõi cần nắm trước khi làm lab

| Khái niệm | Giải thích |
|---|---|
| **MFU** (Model FLOPs Utilization) | % FLOPs thực sự dùng / peak FLOPs. Tốt: 35–50%. Giá trị thực đo hiệu quả tính toán. |
| **MBU** (Model Bandwidth Utilization) | % băng thông HBM thực sự dùng / peak. Dùng cho workload memory-bound (ví dụ: decode). |
| **GPU-Util %** (từ `nvidia-smi`) | Chỉ đo "clock có đang hoạt động không" — **KHÔNG phải hiệu quả**. GPU-Util 98% vẫn có thể MFU 20%. |
| **Cascade** | Định tuyến request đơn giản sang model nhỏ (rẻ hơn 15×), chỉ dùng model lớn khi cần. |
| **Prompt Caching** | Phần input đã cache được chiết khấu 90% (chỉ tính 10% giá). |
| **Batch API** | Nhóm request không cần real-time → chiết khấu 50%. |
| **Spot Instance** | GPU giá thấp nhưng có thể bị thu hồi. Phù hợp với job có thể gián đoạn + checkpoint. |
| **Reserved Instance** | Cam kết 1–3 năm → chiết khấu ~45%. Chỉ có lợi khi utilization ≥ 55%. |
| **Showback** | Hiển thị chi phí theo team để nhận thức, chưa thực sự tính phí. |
| **Chargeback** | Thực sự thu tiền từ team theo mức sử dụng. Cần tag coverage ≥ 80%. |
| **FOCUS** | Chuẩn mở đa nhà cung cấp cho dữ liệu chi phí cloud (FinOps Foundation). |

---

## 2. Cài đặt môi trường

### Bước 2.1 — Clone hoặc vào thư mục lab

```bash
cd Day25-Track2-GPU-FinOps-Lab
```

### Bước 2.2 — Tạo virtual environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

Khi kích hoạt thành công, terminal sẽ hiển thị `(.venv)` ở đầu dòng lệnh.

### Bước 2.3 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Thư viện cần:
- `pandas>=2.0` — xử lý CSV
- `matplotlib>=3.7` — vẽ biểu đồ savings waterfall
- `pytest>=7.4` — chạy unit tests

### Bước 2.4 — Kiểm tra cài đặt

```bash
python verify.py
```

**Kết quả mong đợi:**
```
============================================================
  LAB 25 VERIFY
============================================================
  [PASS] M1 flags the GPU-Util lie (gpu-h100-4)
  [PASS] M1 detects idle waste
  [PASS] M2 $/1M-token drops after optimization
  ...
------------------------------------------------------------
  11/11 checks passed
============================================================
```

> Nếu chưa được 11/11 ngay lúc này, tiếp tục làm lab và chạy lại sau khi hoàn thành.

---

## 3. Khám phá dữ liệu đầu vào

### Bước 3.1 — Sinh dữ liệu tổng hợp

```bash
python data/generate.py
```

Lệnh này tạo 4 file CSV trong `data/` với seed cố định (seed=25), luôn cho kết quả giống nhau.

### Bước 3.2 — Đọc và hiểu từng file

**`data/price_catalog.csv`** — Bảng giá GPU:

```python
import pandas as pd
df = pd.read_csv("data/price_catalog.csv")
print(df[["gpu_type", "on_demand_hr", "spot_hr", "reserved_3yr_hr", "peak_tflops_fp16", "watts"]])
```

Chú ý: H100 on-demand ~$2.5/giờ. Spot rẻ hơn đáng kể. Reserved 3yr rẻ nhất nhưng cần cam kết.

**`data/gpu_telemetry.csv`** — Telemetry GPU theo giờ:

```python
df = pd.read_csv("data/gpu_telemetry.csv")
print(df[["gpu_id", "gpu_type", "gpu_util_pct", "achieved_tflops", "achieved_bw_tbs"]].head(20))
```

Tìm các GPU có `gpu_util_pct` cao nhưng `achieved_tflops` thấp so với peak — đây là "GPU-Util lie".

**`data/token_usage.csv`** — Nhật ký 2,400 request LLM:

```python
df = pd.read_csv("data/token_usage.csv")
print(df[["route_tier", "input_tokens", "output_tokens", "cached_input_tokens", "is_batch", "team"]].head(10))
```

`route_tier` = `"small"` hoặc `"large"`. `is_batch` = 1 nếu request có thể gộp batch.

**`data/workloads.csv`** — 8 job training/inference:

```python
df = pd.read_csv("data/workloads.csv")
print(df[["job_id", "gpu_type", "num_gpus", "hours_per_day", "interruptible"]])
```

`interruptible=1` → phù hợp với spot instance + checkpoint.

---

## 4. Mission 1 — Kiểm toán hiệu quả GPU

**Mục tiêu:** Phát hiện GPU đang "nói dối" (GPU-Util cao nhưng MFU thấp) và tính lãng phí idle.

### Bước 4.1 — Đọc code `finops/metrics.py`

Mở file và đọc kỹ 3 hàm:
- `compute_mfu(achieved_tflops, peak_tflops)` → trả về tỷ lệ 0.0–1.0
- `flag_util_lies(rows, util_threshold=0.90, mfu_threshold=0.30)` → các GPU có util ≥ 90% nhưng MFU < 30%
- `idle_waste_usd(idle_hours, on_demand_hr)` → dollars lãng phí do GPU để trống

### Bước 4.2 — Chạy M1

```bash
python missions/m1_efficiency_audit.py
```

**Kết quả mong đợi (mẫu):**
```
== M1 Efficiency Audit ==
GPU            type   util%    MFU    MBU  idle_h
gpu-h100-4     H100    98.0  0.202  0.450       0
...
GPU-Util LIES (util>=90% but MFU<30%): ['gpu-h100-4']
Idle waste (1 day): $125.00  ->  $3,750/month
```

### Bước 4.3 — Phân tích kết quả

Trả lời các câu hỏi sau (để hiểu sâu):

1. GPU nào có `GPU-Util` cao nhất? MFU của nó là bao nhiêu?
2. Tại sao `GPU-Util 98%` có thể đi kèm với `MFU 20%`? (Gợi ý: GPU "bận" nhưng làm gì? Memory stall? I/O wait?)
3. Lãng phí idle tính ra bao nhiêu `/tháng`? Chiếm bao nhiêu % tổng chi phí?

### Bước 4.4 — Hiểu roofline model (tùy chọn nhưng nên đọc)

```python
from finops.metrics import roofline_regime, arithmetic_intensity
# H100 ridge point: ~295 FLOP/byte (BF16)
# LLM decode: ~1-2 FLOP/byte → memory-bound
# LLM prefill: ~455 FLOP/byte → compute-bound
print(roofline_regime(1.5, 295))   # 'memory-bound'
print(roofline_regime(455, 295))   # 'compute-bound'
```

---

## 5. Mission 2 — Đòn bẩy chi phí Inference

**Mục tiêu:** Hiểu 3 "đòn bẩy" chính để giảm `$/1M-token`:
1. **Cascade** — định tuyến request sang model nhỏ khi đủ
2. **Prompt Caching** — chiết khấu 90% cho phần input đã cache
3. **Batch API** — chiết khấu 50% cho request không cần real-time

### Bước 5.1 — Đọc code `finops/pricing.py`

Tập trung vào:
- `request_cost(input_tok, output_tok, price_in, price_out, cached_in, batch)` — chi phí một request
- `dollars_per_million(total_cost, total_tokens)` — quy về `$/1M-token`
- `discount_stack(batch, cache_hit_frac)` — xem chiết khấu nhân lên nhau thế nào

### Bước 5.2 — Chạy M2

```bash
python missions/m2_inference_levers.py
```

**Kết quả mong đợi (mẫu):**
```
== M2 Inference Cost Levers ==
requests=2400  tokens=1,234,567
baseline  : $45.23/day   $36.643/1M-token
optimized : $12.81/day   $10.374/1M-token
savings   : 71.7%  (cascade + caching + batch)
discount stack (batch + 100% cache): 0.050 of naive
```

### Bước 5.3 — Tính tay để kiểm tra hiểu biết

```python
from finops.pricing import request_cost, discount_stack

# Baseline: model lớn, không cache, không batch
baseline = request_cost(
    input_tok=1000, output_tok=200,
    price_in_per_m=3.00, price_out_per_m=15.00
)
print(f"Baseline: ${baseline:.4f}")

# Optimized: model nhỏ + cache 80% input + batch
optimized = request_cost(
    input_tok=1000, output_tok=200,
    price_in_per_m=0.20, price_out_per_m=0.40,
    cached_in=800, batch=True
)
print(f"Optimized: ${optimized:.4f}")

# Discount stack: batch=True + 80% cache hit
print(f"Effective fraction: {discount_stack(batch=True, cache_hit_frac=0.8):.3f}")
```

### Bước 5.4 — Phân tích kết quả

1. Trong 3 đòn bẩy (cascade, cache, batch), đòn bẩy nào đóng góp savings lớn nhất?
2. Tại sao `discount_stack(batch=True, cache_hit_frac=1.0) = 0.05`? (Gợi ý: 50% × 10% = ?)
3. Khi nào **không nên** dùng batch API? (Gợi ý: nghĩ về latency)

---

## 6. Mission 3 — Chiến lược mua GPU

**Mục tiêu:** Chọn đúng tier (on-demand / spot / reserved) cho từng workload để tối ưu chi phí.

### Bước 6.1 — Hiểu logic `recommend_tier()`

```python
# Trong finops/pricing.py:
def recommend_tier(hours_per_day, interruptible, reserved_discount=0.45):
    duty = hours_per_day / 24.0
    be = break_even_utilization(reserved_discount)  # = 1 - 0.45 = 55%
    if interruptible and hours_per_day < 24:
        return "spot"       # Có thể gián đoạn → spot + checkpoint
    if duty >= be:
        return "reserved"   # Duty cycle ≥ 55% → reserved
    return "on_demand"      # Còn lại
```

**Điểm hòa vốn:** Để reserved có lợi hơn on-demand, cần utilization ≥ `1 - discount`. Với 45% discount → cần ≥ 55% = 13.2 giờ/ngày.

### Bước 6.2 — Chạy M3

```bash
python missions/m3_purchasing.py
```

**Kết quả mong đợi (mẫu):**
```
== M3 Purchasing Strategy ==
break-even utilization @ 45% reserved discount = 55%
job                gpu    tier        on-demand    optimized
training-001       H100   spot          $15,000      $7,200
inference-001      H100   reserved      $10,800      $5,940
...
monthly: on-demand $45,000 -> optimized $25,200  (44.0% saved)
```

### Bước 6.3 — Tìm hiểu spot checkpoint simulation

```python
from finops.pricing import spot_checkpoint_cost

result = spot_checkpoint_cost(
    job_hours=720,          # 30 ngày × 24h
    spot_hr=1.40,           # Giá spot H100
    on_demand_hr=2.50,      # Giá on-demand H100
    interrupt_rate=0.05,    # 5% khả năng bị thu hồi mỗi giờ
    ckpt_overhead_frac=0.03 # 3% overhead để ghi checkpoint
)
print(result)
# {'spot_effective_hours': 759.6, 'spot_cost': 1063.44, 'on_demand_cost': 1800.0, 'savings_pct': 40.9}
```

### Bước 6.4 — Phân tích kết quả

1. Job nào được đề xuất dùng spot? Tại sao?
2. Với spot, "effective hours" cao hơn "job hours" thực tế — điều này nghĩa là gì?
3. Job nào được đề xuất reserved nhưng bạn nghĩ on-demand phù hợp hơn (hoặc ngược lại)? Tại sao?

---

## 7. Mission 4 — Phân bổ chi phí

**Mục tiêu:** Chuyển hóa đơn GPU chung thành trách nhiệm theo từng team — từ "visibility" → "showback" → "chargeback".

### Bước 7.1 — Hiểu lý thuyết

**Thang trưởng thành:**
```
Visibility → Showback → Chargeback
   (thấy)    (thông báo)  (thu tiền)
```

Chỉ có thể chargeback khi **tag coverage ≥ 80%** — nếu không, bạn đang tính phí dựa trên dữ liệu không đầy đủ.

### Bước 7.2 — Chạy M4

```bash
python missions/m4_allocation.py
```

**Kết quả mong đợi (mẫu):**
```
== M4 Cost Allocation ==
cost by team ($/day):
  ml-research    $    8.45
  platform       $    3.12
  product        $    1.24
tag coverage: 92%  ->  chargeback ready? True
FOCUS export -> outputs/focus_export.csv (50 rows)
```

### Bước 7.3 — Kiểm tra FOCUS export

```bash
head -5 outputs/focus_export.csv
```

**Cột của FOCUS:**
- `BillingAccountId` — tài khoản thanh toán
- `ChargePeriodStart` — kỳ tính phí
- `ServiceCategory` — loại dịch vụ (`AI and Machine Learning`)
- `BilledCost` — chi phí thực tế tính
- `team`, `project` — tags phân bổ

### Bước 7.4 — Phân tích kết quả

1. Team nào tốn nhiều nhất? Tỷ lệ so với tổng là bao nhiêu?
2. Tag coverage là bao nhiêu %? Có đủ để chargeback không?
3. Tại sao FOCUS lại quan trọng khi công ty dùng nhiều cloud provider?

---

## 8. Mission 5 — Báo cáo tổng hợp

**Mục tiêu:** Gộp kết quả từ M1–M4, tạo báo cáo baseline vs. optimized với biểu đồ waterfall.

### Bước 8.1 — Hiểu cấu trúc M5

M5 kết hợp 4 "đòn bẩy" tiết kiệm:

| Đòn bẩy | Nguồn |
|---|---|
| Inference (cascade/cache/batch) | M2 |
| Purchasing (spot/reserved) | M3 |
| Right-size util-lies | M1 — hạ cấp GPU bị "lie" xuống tier thấp hơn |
| Kill idle GPUs | M1 — tắt GPU chạy không, utilization < 10% |

### Bước 8.2 — Chạy M5

```bash
python missions/m5_report.py
```

**Kết quả mong đợi:**
```
== M5 Optimization Report ==
# NimbusAI — GPU Cost Optimization Report
Baseline spend: $27,133
Optimized spend: $14,626
Projected savings: $12,507 (46%)
...
Written: outputs/report.md + outputs/savings.png
```

### Bước 8.3 — Đọc báo cáo đầu ra

```bash
cat outputs/report.md
```

Mở `outputs/savings.png` để xem biểu đồ waterfall — mỗi cột là một đòn bẩy tiết kiệm.

### Bước 8.4 — Kiểm tra section Sustainability

Báo cáo có phần:
```markdown
## Sustainability
- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1
```

Vùng **europe-north1** (Na Uy) rẻ và sạch nhất do thủy điện. Vùng **europe-central2** (Ba Lan) có carbon 660 gCO2/kWh — dơ nhất. Chọn vùng triển khai đúng vừa tiết kiệm tiền, vừa giảm phát thải.

---

## 9. Chạy toàn bộ kiểm tra

### Bước 9.1 — Chạy tất cả missions cùng lúc

```bash
python missions/run_all.py
```

### Bước 9.2 — Chạy verify.py

```bash
python verify.py
```

Phải đạt **11/11 checks passed**. Nếu fail ở check nào, đọc chi tiết error để biết mission nào cần sửa.

**Các check quan trọng:**
- `M1 flags the GPU-Util lie (gpu-h100-4)` — `flag_util_lies()` phải trả về đúng GPU
- `M2 inference savings in 60-95% band` — savings phải trong khoảng 60–95%
- `M3 recommends a spot tier` và `M3 recommends a reserved tier` — phải có cả hai
- `M4 tag coverage 85-100%` — coverage ≥ 85%
- `M5 total savings in 40-95% band` — tổng savings phải đạt 40–95%
- `M5 report.md written` — file phải tồn tại

### Bước 9.3 — Chạy pytest

```bash
pytest -q
```

**Kết quả mong đợi:**
```
...............
15 passed in 0.42s
```

Nếu có test fail, đọc error message, tìm file test tương ứng trong `tests/` để hiểu assertion nào không thỏa.

---

## 10. Phần mở rộng "Your Turn"

Cần thực hiện **ít nhất 2 trong 5** phần mở rộng để đạt điểm 20% còn lại.

### Extension 1 — Cải thiện `recommend_tier()`

**File cần sửa:** `finops/pricing.py` — hàm `recommend_tier()`

```python
# Thêm logic cân nhắc:
# 1. Interruption rate khác nhau theo GPU type (H100 spot ít bị thu hồi hơn A10G)
# 2. So sánh 3yr reserved vs 1yr reserved theo duration thực tế của job
def recommend_tier(hours_per_day, interruptible, reserved_discount=0.45,
                   gpu_type=None, job_days=None):
    # TODO: viết lại logic tại đây
    ...
```

**Đo lường:** Chạy lại M3, so sánh `savings_pct` trước và sau khi sửa.

### Extension 2 — Right-sizing theo MBU

**File cần sửa:** `missions/m1_efficiency_audit.py`

Trong M1, thêm logic: với các GPU memory-bound (MBU thấp), gợi ý GPU nào rẻ hơn có cùng MBU target.

```python
# Tính $/GB-VRAM cho từng GPU type
# So sánh peak_bw_tbs để tìm GPU phù hợp hơn cho inference
```

### Extension 3 — `cache_is_worth_it()`

**File cần tạo hoặc sửa:** `finops/pricing.py`

```python
def cache_is_worth_it(
    avg_cache_reads: float,     # Lần đọc lại trung bình mỗi cached prefix
    write_cost_per_m: float,    # Chi phí ghi cache (Gemini tính tiền lưu trữ)
    read_discount: float = 0.10 # 10% = 90% off
) -> bool:
    """Cache chỉ tiết kiệm tiền khi tổng tiết kiệm từ đọc > chi phí ghi."""
    # Break-even: đủ số lần đọc để bù chi phí ghi
    ...
```

### Extension 4 — Ngân sách Reasoning

**File cần sửa:** `missions/m2_inference_levers.py` và `missions/m5_report.py`

```python
# Trong token_usage.csv có cột is_reasoning
# Tính riêng chi phí $ và Wh cho is_reasoning=1
# So sánh với is_reasoning=0 cùng số token
# Đề xuất: giới hạn reasoning chỉ khi confidence score < threshold
```

Dùng `finops.sustainability.wh_per_query(tokens, is_reasoning=True)` — tiêu thụ năng lượng gấp ~80× query thông thường.

### Extension 5 — Carbon-aware Scheduling

**File cần sửa:** `missions/m3_purchasing.py` hoặc tạo file mới

```python
from finops.sustainability import REGION_CARBON, carbon_g, wh_per_query

# Với mỗi job interruptible=1 trong workloads.csv:
# 1. Tính carbon hiện tại (giả sử chạy ở us-east-1)
# 2. Tính carbon nếu chạy ở vùng sạch nhất (europe-north1)
# 3. Báo cáo: gCO2e tiết kiệm được và % giảm
```

---

## 11. Chuẩn bị nộp bài

### Checklist trước khi nộp

```
[ ] python verify.py  →  11/11 checks passed
[ ] pytest -q         →  15 passed
[ ] outputs/report.md tồn tại và có đủ các section
[ ] outputs/savings.png tồn tại (nếu matplotlib hoạt động)
[ ] outputs/focus_export.csv tồn tại
[ ] Đã thực hiện ≥2 extension với kết quả đo lường
```

### Bài viết ngắn đi kèm (write-up)

Viết 1–2 trang (không có template cứng) trả lời:

1. **Baseline vs. Optimized:** Chi phí trước và sau, `$/1M-token` trước và sau. Tiết kiệm tổng cộng bao nhiêu %?
2. **Phân tích từng đòn bẩy:** Đòn bẩy nào đóng góp nhiều nhất? Tại sao?
3. **GPU-Util Lie:** GPU nào bị "lie"? Tác động tài chính là gì?
4. **Phần mở rộng đã làm:** Mô tả từng extension, kết quả đo được, insight quan trọng nhất.
5. **Khuyến nghị cho NimbusAI:** Nếu bạn là FinOps lead, 3 hành động đầu tiên bạn sẽ làm là gì?

### File cần nộp

```
outputs/report.md
outputs/savings.png
outputs/focus_export.csv
writeup.md
```

---

## 12. Các lỗi thường gặp

### Lỗi: `ModuleNotFoundError: No module named 'pandas'`

```bash
# Kiểm tra virtual environment đã được kích hoạt chưa
which python   # Phải trỏ vào .venv/bin/python
pip install -r requirements.txt
```

### Lỗi: `FileNotFoundError: data/gpu_telemetry.csv`

```bash
python data/generate.py   # Sinh lại dữ liệu trước
```

### Lỗi: verify.py fail ở M2 savings out of band

Kiểm tra `finops/pricing.py` — hàm `request_cost()` có đang áp dụng đúng `cache_discount=0.10` và `batch_discount=0.50` không?

### Lỗi: pytest fail `test_flag_util_lies`

Hàm `flag_util_lies()` cần:
- Nhận `gpu_util_pct` theo thang 0–100
- Nhưng so sánh nội bộ chia 100 (`util = float(r["gpu_util_pct"]) / 100.0`)
- Ngưỡng: `util >= 0.90` VÀ `mfu < 0.30`

### Lỗi: matplotlib không tạo được PNG

Lab vẫn pass nếu không có matplotlib (PNG bị bỏ qua). Kiểm tra `pip show matplotlib` và cài lại nếu cần.

### Warnings về `is_batch` type

Nếu thấy warning về kiểu dữ liệu bool, kiểm tra:
```python
is_batch = bool(int(num(r["is_batch"])))   # Không phải bool(r["is_batch"]) trực tiếp
```

---
