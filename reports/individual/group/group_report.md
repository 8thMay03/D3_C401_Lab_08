# Báo Cáo Nhóm — Lab Day 08: Full RAG Pipeline

**Tên nhóm:** D3-C401 — Lab 08 (CS + IT Helpdesk RAG)  
**Thành viên:**
| Tên | Vai trò | Email |
|-----|---------|-------|
| Nguyễn Quốc Khánh | Tech Lead | — |
| Lý Quốc An | Tech Lead | — |
| Nguyễn Quang Minh | Eval Owner | — |
| Lưu Thị Ngọc Quỳnh | Documentation Owner | — |
| Nguyễn Bá Khánh | Retrieval Owner | — |
| Nguyễn Phương Nam | Retrieval Owner | — |
| Lưu Quang Lực | Eval Owner | — |
| Đinh Văn Thư | Eval Owner | — |

**Ngày nộp:** 13/04/2026  
**Repo:** `D3_C401_Lab_08`  
**Độ dài khuyến nghị:** 600–900 từ

---

> **Hướng dẫn nộp group report:**
>
> - File này nộp tại: `reports/group_report.md`
> - Deadline: Được phép commit **sau 18:00** (xem SCORING.md)
> - Tập trung vào **quyết định kỹ thuật cấp nhóm** — không trùng lặp với individual reports
> - Phải có **bằng chứng từ code, scorecard, hoặc tuning log** — không mô tả chung chung

---

## 1. Pipeline nhóm đã xây dựng (150–200 từ)

> Mô tả ngắn gọn pipeline của nhóm:
> - Chunking strategy: size, overlap, phương pháp tách (by paragraph, by section, v.v.)
> - Embedding model đã dùng
> - Retrieval mode: dense / hybrid / rerank (Sprint 3 variant)

**Chunking decision:**
> VD: "Nhóm dùng chunk_size=500, overlap=50, tách theo section headers vì tài liệu có cấu trúc rõ ràng."

Nhóm dùng **section-aware chunking**: ưu tiên tách theo heading/section để giữ nguyên cấu trúc điều khoản, sau đó mới cắt theo độ dài để tránh “cắt giữa điều kiện/ngoại lệ”. Cấu hình chạy chính: **chunk_size ≈ 400 tokens**, **overlap ≈ 80 tokens**. Lý do: 400 tokens đủ giữ trọn 1 cụm ý của policy/SOP; overlap 80 giảm rủi ro mất ý ở ranh giới chunk; đồng thời phù hợp với prompt grounded (chỉ chọn top-3 chunk) để tránh context quá dài.

**Embedding model:**

Thiết kế hỗ trợ 2 chế độ embedding để dễ demo theo môi trường:
- Có API key: OpenAI `text-embedding-3-small`
- Không có API key: SentenceTransformer `paraphrase-multilingual-MiniLM-L12-v2` (384 chiều)  
Trong lần chạy scorecard hiện tại, collection Chroma có dimension **384** (nhánh local), nên toàn bộ index/query dùng cùng backend để tránh mismatch dimension.

**Retrieval variant (Sprint 3):**
> Nêu rõ variant đã chọn (hybrid / rerank / query transform) và lý do ngắn gọn.

Nhóm thử **hybrid + rerank** cho Sprint 3: dense (Chroma) + sparse (BM25) được gộp bằng RRF, sau đó rerank bằng cross-encoder trước khi chọn top-3 vào prompt. Lý do: corpus có cả câu tự nhiên lẫn keyword/mã định danh (SLA, Level 3, mã lỗi), nên hybrid giúp “bắt keyword” tốt hơn; rerank kỳ vọng giảm noise khi dense/sparse trả về nhiều đoạn gần nghĩa.

---

## 2. Quyết định kỹ thuật quan trọng nhất (200–250 từ)

> Chọn **1 quyết định thiết kế** mà nhóm thảo luận và đánh đổi nhiều nhất trong lab.
> Phải có: (a) vấn đề gặp phải, (b) các phương án cân nhắc, (c) lý do chọn.

**Quyết định:** Chọn **baseline dense** làm cấu hình mặc định thay vì “hybrid + rerank” dù hybrid nghe có vẻ mạnh hơn về mặt lý thuyết.

**Bối cảnh vấn đề:**

Trong Sprint 3, nhóm đứng trước câu hỏi: nên “nâng cấp retrieval” sang hybrid + rerank để tăng khả năng tìm đúng context cho query có keyword/alias hay không. Mục tiêu không chỉ là recall (lấy đúng tài liệu) mà còn là **độ đúng trọng tâm và đầy đủ** của câu trả lời khi chỉ đưa **top-3 chunk** vào prompt grounded. Thực tế, pipeline đã có context recall cao (retrieve đúng tài liệu), nhưng completeness vẫn là điểm yếu ở một số câu có điều kiện/ngoại lệ hoặc cần liệt kê đầy đủ.

**Các phương án đã cân nhắc:**

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|-----------|
| Dense (Chroma) | Đơn giản, ổn định, ít “quá tay” với dữ liệu nhỏ; dễ debug | Dễ bỏ lỡ exact keyword/alias nếu embedding không kéo đủ gần |
| Hybrid (Dense + BM25) + Rerank | Bắt keyword tốt hơn; kỳ vọng chọn context “đúng nhất” trước khi vào prompt | Dễ overfit vào tín hiệu gần nghĩa/keyword, bỏ sót điều kiện phụ; thêm độ phức tạp và failure mode |

**Phương án đã chọn và lý do:**

Nhóm chọn **baseline dense** làm default cho demo/submit vì cho chất lượng tổng thể ổn định hơn trong bộ grading questions nhỏ. Hybrid + rerank vẫn giữ lại như một “variant để thử nghiệm” (tính năng Sprint 3) vì nó hữu ích khi gặp query thuần keyword, nhưng không được dùng làm cấu hình mặc định nếu không chứng minh được cải thiện trên scorecard.

**Bằng chứng từ scorecard/tuning-log:**

Dựa trên scorecard đã chạy ngày 13/04/2026:
- `results/scorecard_baseline.md`: Faithfulness **4.30/5**, Relevance **4.50/5**, Context Recall **5.00/5**, Completeness **3.40/5**
- `results/scorecard_variant.md` (hybrid + rerank): Faithfulness **4.60/5** (tăng), Relevance **4.40/5** (giảm nhẹ), Context Recall **5.00/5** (giữ nguyên), Completeness **3.40/5** (không cải thiện)  
Kết luận: variant không tăng completeness và còn giảm nhẹ relevance, nên baseline được chọn làm cấu hình mặc định.

---

## 3. Kết quả grading questions (100–150 từ)

> Sau khi chạy pipeline với grading_questions.json (public lúc 17:00):
> - Câu nào pipeline xử lý tốt nhất? Tại sao?
> - Câu nào pipeline fail? Root cause ở đâu (indexing / retrieval / generation)?
> - Câu gq07 (abstain) — pipeline xử lý thế nào?

**Ước tính điểm raw:** ~85 / 98 (ước lượng theo trung bình scorecard \(\approx 0.87 \times 98\))

**Câu tốt nhất:** ID: `gq03` (Refund) — Lý do: evidence rõ ràng, retrieval trả đúng policy; answer bám sát context và đủ ý chính nên đồng thời đạt điểm cao ở faithfulness/relevance/recall/completeness.

**Câu fail:** ID: `gq07` (Insufficient Context) — Root cause: **abstain/gating chưa chặt**. Scorecard cho thấy recall = `None` và faithfulness/relevance rất thấp (1/5), nghĩa là pipeline vẫn “trả lời” dù không có đủ context; đây là lỗi ở lớp generation/gating.

**Câu gq07 (abstain):** Cả baseline và variant đều bị chấm thấp vì không abstain đúng cách. Kỳ vọng đúng: trả lời “không đủ dữ liệu/không có trong tài liệu”, không suy diễn, và không gắn citation giả.

---

## 4. A/B Comparison — Baseline vs Variant (150–200 từ)

> Dựa vào `docs/tuning-log.md`. Tóm tắt kết quả A/B thực tế của nhóm.

**Biến đã thay đổi (chỉ 1 biến):** Retrieval stack — từ **dense** sang **hybrid (dense+BM25, RRF fusion) + cross-encoder rerank**.

| Metric | Baseline | Variant | Delta |
|--------|---------|---------|-------|
| Faithfulness | 4.30/5 | 4.60/5 | +0.30 |
| Relevance | 4.50/5 | 4.40/5 | -0.10 |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.40/5 | 3.40/5 | 0.00 |

**Kết luận:**
> Variant tốt hơn hay kém hơn? Ở điểm nào?

Variant hybrid + rerank **không tốt hơn baseline** trên mục tiêu “trả lời đúng trọng tâm và đủ ý”: completeness **không tăng**, relevance **giảm nhẹ**, trong khi recall vốn đã tối đa (5.00/5) nên khó còn dư địa. Điểm sáng là faithfulness tăng, nhưng do completeness không cải thiện và vẫn còn failure mode `gq07`, nhóm giữ baseline dense làm default và coi variant là hướng thử nghiệm để tối ưu cho các query nặng keyword/alias trong tương lai.

---

## 5. Phân công và đánh giá nhóm (100–150 từ)

> Đánh giá trung thực về quá trình làm việc nhóm.

**Phân công thực tế:**

| Thành viên | Phần đã làm | Sprint |
|------------|-------------|--------|
| Nguyễn Quốc Khánh | Nối pipeline end-to-end, chạy eval/scorecard, tổng hợp kết quả | 2, 4 |
| Lý Quốc An | Chunking + metadata + retrieval (dense/hybrid/rerank), rà soát failure mode | 1, 3 |
| Nguyễn Quang Minh | Chuẩn hóa bộ câu hỏi/chấm điểm, xuất scorecard baseline/variant | 4 |
| Lưu Thị Ngọc Quỳnh | Viết/chuẩn hóa `docs/architecture.md`, hỗ trợ group report | 4 |
| Nguyễn Bá Khánh | Bổ sung tuning-log, tổng hợp nhận xét A/B | 1, 4 |
| Nguyễn Phương Nam | Rà soát chất lượng câu trả lời, ghi nhận fail cases (gq07) | 3, 4 |
| Lưu Quang Lực | Hỗ trợ documentation và rà soát formatting | 4 |

**Điều nhóm làm tốt:**

Nhóm tuân thủ A/B rule (mỗi lần đổi một biến), và dùng scorecard để quyết định dựa trên dữ liệu thay vì “cảm giác”. Việc ghi `architecture.md` giúp trace được nguyên nhân khi metric không cải thiện (ví dụ recall cao nhưng completeness thấp).

**Điều nhóm làm chưa tốt:**

Abstain/gating cho câu “insufficient context” chưa robust (thể hiện rõ ở `gq07`). Ngoài ra, phần tuning-log có chỗ chưa nhất quán (mô tả biến thay đổi và ngày chạy), nên cần chuẩn hóa lại quy trình ghi log để tránh hiểu sai khi review.

---

## 6. Nếu có thêm 1 ngày, nhóm sẽ làm gì? (50–100 từ)

> 1–2 cải tiến cụ thể với lý do có bằng chứng từ scorecard.

Nhóm sẽ ưu tiên 2 cải tiến có tác động trực tiếp lên scorecard:
1) **Fix abstain**: thêm ngưỡng “context đủ mạnh” (ví dụ min similarity / min rerank score, hoặc yêu cầu có citation) trước khi trả lời, để xử lý đúng `gq07` và các câu không có evidence.  
2) **Tăng completeness có kiểm soát**: thử tăng `top_k_select` (3 → 4) hoặc thêm bước “answer checklist” trong prompt cho các câu policy nhiều điều kiện/ngoại lệ, rồi chạy lại A/B để xem completeness có tăng mà không làm faithfulness giảm.

---

*File này lưu tại: `reports/group_report.md`*  
*Commit sau 18:00 được phép theo SCORING.md*
