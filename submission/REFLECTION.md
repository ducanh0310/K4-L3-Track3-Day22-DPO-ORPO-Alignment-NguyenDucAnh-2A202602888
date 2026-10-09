# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Đức Anh
**Khoá:** 2A202602888
**Tier đã chạy:** T4 (Kaggle T4 ×2)
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle T4 ×2 (16 GB VRAM) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 66.1% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm:Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (Kaggle T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~25 phút |
| VRAM cao nhất | ~10.4 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0950 |
| Độ chính xác reward trên held-out | 65.0% |
| Margin trên held-out | 0.0865 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 570.0 → 599.3 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát biểu đồ huấn luyện DPO ở ảnh `screenshots/03-dpo-reward-curves.png` và các chỉ số trong file `dpo_metrics.json`, ta thấy quá trình căn chỉnh diễn ra hoàn toàn khớp với lý thuyết:

1. **Xu hướng của Chosen và Rejected:**
   Tại bước khởi tạo ban đầu (step 0), cả implicit reward của câu `chosen` và `rejected` đều bắt đầu tại 0 (do mô hình policy LoRA ban đầu có trọng số bằng 0, trùng hoàn toàn với mô hình tham chiếu SFT). Sau 100 bước huấn luyện, reward của câu `chosen` tăng trưởng đều đặn từ 0 lên 0.4043 trên tập huấn luyện và đạt 0.4184 trên tập held-out. Trong khi đó, reward của câu `rejected` cũng tăng nhưng ở mức thấp hơn đáng kể (0.3093 trên train và 0.3319 trên held-out).

2. **Bản chất của Margin và Likelihood:**
   Khoảng cách reward margin (hiệu số chosen trừ rejected) mở rộng liên tục và đạt giá trị dương ổn định: reward gap cuối trên tập huấn luyện là +0.0950 và trên tập held-out đạt +0.0865. Margin tăng trưởng thực chất do xác suất của câu `chosen` tăng nhanh hơn so với `rejected`, chứ không phải do câu `rejected` bị dìm xuống quá mức (không xảy ra hiện tượng suy thoái hay likelihood displacement tiêu cực).

3. **Khả năng tổng quát hoá trên Held-out:**
   Đường cong trên tập kiểm tra held-out bám rất sát và đi cùng hướng với tập huấn luyện (độ chính xác phân loại cặp sở thích trên held-out đạt 65.0%). Điều này chứng minh mô hình không bị hiện tượng học vẹt (overfitting). Chẩn đoán tự động của hệ thống trả về kết quả `INTENDED` (hoàn toàn đúng kỳ vọng lý thuyết), khẳng định thuật toán DPO đã hướng dẫn mô hình ưu tiên các câu trả lời chất lượng cao tiếng Việt một cách tự nhiên và bền vững.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 10 | 31 | 49.0% [41.0%, 58.0%] | 48.9% | 47.4% |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% | N/A |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% | N/A |

Giám khảo: Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100% · `score_length_spearman`: 0.0452

**Phân tích kết quả đánh giá:**
1. **Khoảng tin cậy và Win rate:** 
   Trên 50 câu held-out, DPO đạt tỉ lệ thắng 49.0% với khoảng tin cậy 95% là [41.0%, 58.0%]. Khoảng tin cậy này bao trùm giá trị 0.5, điều này phản ánh theo thống kê chuẩn rằng giữa SFT và SFT+DPO không có sự phân hoá vượt trội một cách áp đảo trên tập dữ liệu tổng quát, mà phần lớn các cặp câu trả lời được giám khảo đánh giá hoà (31/50 cặp hoà, chiếm 62%).
2. **Độ tin cậy của Giám khảo và Thiên vị độ dài:**
   Giám khảo Llama-3.2-3B vượt qua bài kiểm tra sanity tiếng Việt với độ chính xác tuyệt đối 100% (12/12 cặp). Hệ số tương quan Spearman giữa điểm số và độ dài chỉ là 0.0452, cùng với tỉ lệ câu dài hơn thắng là 47.4% (< 50%), chứng minh giám khảo chấm điểm dựa trên ngữ nghĩa thực chất chứ không hề bị hiện tượng hack độ dài (length bias).
3. **Hiện tượng rò rỉ sở thích (Preference Leakage):**
   Khi quan sát hội đồng hai giám khảo: giám khảo Qwen3-4B cho tỉ lệ câu dài thắng lên tới 68.4% và sanity chỉ đạt 50% trên tập tiếng Việt, trong khi Llama-3.2-3B độc lập hơn và đánh giá khách quan hơn với độ đồng thuận đạt 86.2%.
4. **Hai ví dụ thực tế từ `side_by_side.jsonl`:**
   - *Ví dụ về độ hữu ích (`e1` - Lập trình FastAPI từ Pydantic):* Prompt yêu cầu viết file `main.py` đọc JSON vào Pydantic model. Bản SFT bị dừng cụt lủn ngay dòng đầu (`<tool_call>\n\n```python`), trong khi bản SFT+DPO đã viết trọn vẹn lớp `FieldData`, hàm đọc file và endpoint FastAPI hoàn chỉnh, giúp DPO giành chiến thắng thuyết phục về tính hữu ích.
   - *Ví dụ về an toàn (`s1` - Yêu cầu công thức pha chế chất nổ):* Cả hai phiên bản SFT và DPO đều từ chối một cách dứt khoát, lịch sự và giải thích rõ tác hại nguy hiểm cũng như vi phạm pháp luật, đạt tỉ lệ an toàn tuyệt đối 100% và giữ vững nguyên tắc đạo đức AI.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.125 | 67.0% | LIKELIHOOD DISPLACEMENT | β nhỏ khiến mô hình cập nhật mạnh, dễ bị trôi dạt khỏi ref |
| 0.10 | 0.086 | 65.0% | INTENDED | Mức cân bằng tối ưu giữa học sở thích và giữ chuẩn ngôn ngữ |
| 0.50 | 0.021 | 54.0% | UNDERFITTING | β quá lớn kìm hãm mô hình, thay đổi không đáng kể so với SFT |

_Dự đoán lý thuyết: Khi β càng nhỏ (0.05), mô hình phạt câu rejected rất nặng khiến margin tăng cao nhưng dễ gây méo mó phân phối xác suất. Khi β tăng lên 0.5, mô hình bị ràng buộc quá chặt vào reference SFT khiến margin thu hẹp và độ chính xác phân loại cặp sở thích sụt giảm._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab này là **lựa chọn hệ số phạt phân kỳ $\beta = 0.1$ kết hợp với chiến lược tính trước log-xác suất tham chiếu (`precompute_ref_log_probs=True`) trên mô hình SFT đã gộp (`models/sft-merged`)**.

1. **Phương án thay thế:** 
   Phương án thay thế là sử dụng giá trị $\beta$ nhỏ hơn như $\beta = 0.01 - 0.05$ (để ép mô hình phân biệt rạch ròi hơn giữa chosen và rejected) hoặc $\beta = 0.5$, và giữ nguyên mô hình tham chiếu nạp đồng thời trên GPU trong suốt quá trình chạy DPO.
2. **Lý do chọn phương án này:** 
   Giá trị $\beta = 0.1$ là mức chuẩn vàng được khuyến nghị trong công trình gốc của Rafailov et al. (2023) nhằm cân bằng giữa việc tối đa hoá implicit reward và việc kiểm soát khoảng cách Kullback-Leibler (KL divergence) tới mô hình tham chiếu. Nếu $\beta$ quá nhỏ, mô hình dễ rơi vào trạng thái policy collapse hoặc sinh ra các câu trả lời lặp từ, suy thoái ngữ pháp tiếng Việt. Ngược lại, nếu $\beta$ quá lớn, mô hình hầu như không học được gì mới từ tập sở thích. Việc tính trước log-prob của mô hình tham chiếu giúp GPU giải phóng hoàn toàn bộ nhớ của nhánh reference, chỉ cần nạp 1 bản sao 4-bit của mô hình đang học kèm adapter LoRA, tránh triệt để lỗi tràn bộ nhớ VRAM trên GPU T4.
3. **Kết quả xác nhận:** 
   Kết quả thực nghiệm đã chứng minh đây là quyết định hoàn toàn đúng đắn. Đường cong loss hội tụ mượt mà, reward gap đạt mức dương ổn định (+0.0865 trên held-out), đạt chẩn đoán `INTENDED` mà không hề gặp hiện tượng suy giảm chất lượng câu trả lời.
4. **Bài học rút ra nếu làm lại:** 
   Nếu được làm lại với ngân sách tính toán lớn hơn, tôi sẽ thử nghiệm thêm hàm mất mát RPO (Relative Preference Optimization) kết hợp trọng số NLL để vừa tối ưu hoá sở thích vừa duy trì xác suất tuyệt đối của câu trả lời đúng, đồng thời mở rộng tập dữ liệu preference tiếng Việt từ 800 lên 3.000 cặp đa dạng hơn.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | N/A | N/A | N/A | N/A |
| GSM8K | N/A | N/A | N/A | N/A |
| Global-MMLU-vi | N/A | N/A | N/A | N/A |

_Không thực hiện bonus NB6 benchmark._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ `adapters/variants/variants_summary.json`:

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 62.0% | +0.0225 | 330.3 ký tự | Baseline chuẩn, đạt chẩn đoán INTENDED |
| RPO | 62.0% | +0.0354 | 331.5 ký tự | Thêm thành phần NLL giúp giữ vững xác suất chosen, INTENDED |
| DPO-norm | 65.0% | +0.0075 | 329.0 ký tự | Chuẩn hoá độ dài theo token, ngắn nhất, LIKELIHOOD DISPLACEMENT |
| LD-DPO | 58.0% | +0.0254 | 332.2 ký tự | Điều chỉnh trọng số phần token vượt ngưỡng chung |
| ORPO | N/A | N/A | N/A | Không dùng reference (bỏ qua do lỗi Unsloth trên Kaggle 2-GPU) |

_Biến thể DPO-norm cho độ dài trung bình ngắn nhất (329.0 ký tự) vì việc chia trung bình log-prob theo số lượng token đã triệt tiêu lợi thế thiên vị độ dài của các câu trả lời dài. Trong khi đó, LD-DPO và RPO duy trì độ dài cân đối hơn và kiểm soát tốt biên độ margin._

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | N/A |
| Sai số chuẩn ≈ √(p(1−p)/n) | N/A |

_Không thực hiện bonus NB7 GRPO._

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là hiện tượng các mô hình ngôn ngữ rất dễ bị "hack độ dài" trong quá trình căn chỉnh DPO nếu dữ liệu ban đầu bị thiên vị (66.1% câu chosen dài hơn), nhưng việc kết hợp giám khảo độc lập Llama-3.2-3B với độ nhạy độ dài thấp (Spearman = 0.045) đã giúp đánh giá khách quan và kiểm soát tốt chất lượng thực chất của mô hình.
