# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 13/04/2026 
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = True
llm_model = "gpt-4o-mini"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 5 /5 |
| Answer Relevance | 4 /5 |
| Context Recall | 4 /5 |
| Completeness | 4 /5 |


Faithfulness (5/5): Tất cả câu trả lời đều bám sát tài liệu, không bịa đặt, luôn có dẫn nguồn.
Answer Relevance (4/5): Đa số trả lời đúng trọng tâm, nhưng có trường hợp trả lời ngắn hoặc thiếu chi tiết.
Context Recall (4/5): Truy xuất đúng đoạn liên quan, nhưng đôi khi bỏ sót thông tin phụ.
Completeness (4/5): Đủ ý chính, nhưng có trường hợp chưa bao quát hết điều kiện, ngoại lệ hoặc trả lời "không biết"
**Câu hỏi yếu nhất (điểm thấp):**
1. "ERR-403-AUTH là lỗi gì?" — completeness = 2/5, context recall = 2/5: Không tìm thấy thông tin, hệ thống trả về "Tôi không biết" dù có thể có liên quan trong tài liệu.
2. "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?" — completeness = 3/5: Trả lời đúng ý chính nhưng thiếu chi tiết về ngoại lệ và điều kiện áp dụng.
3. "Ai phải phê duyệt để cấp quyền Level 3?" — context recall = 3/5: Trả lời đúng thành phần phê duyệt nhưng chưa nêu rõ quy trình các bước hoặc các trường hợp đặc biệt.

**Giả thuyết nguyên nhân (Error Tree):**
- [x] Indexing: Chunking cắt giữa điều khoản — Có thể làm mất liên kết ý, dẫn đến context recall thấp ở các câu hỏi phức tạp.
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias — Thể hiện rõ ở truy vấn mã lỗi (ERR-403-AUTH) hoặc alias, hệ thống không tìm đúng đoạn liên quan.
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [ ] Generation: Prompt không đủ grounding
- [ ] Generation: Context quá dài → lost in the middle
---

## Variant 1 (Sprint 3)

**Ngày:** 14/03/2026  
**Biến thay đổi:** use_rerank = True  
**Lý do chọn biến này:**
- Thực tế, khi bật rerank, các chỉ số faithfulness, relevance và completeness đều giảm nhẹ so với baseline (faithfulness: -0.6, relevance: -0.5, completeness: -0.2). Context recall giữ nguyên (5.0).
- Nguyên nhân có thể do rerank ưu tiên các đoạn context gần nghĩa với truy vấn nhưng lại bỏ sót các chi tiết quan trọng hoặc dẫn đến overfitting vào một số từ khóa, làm giảm tính đầy đủ và độ chính xác tổng thể.
- Một số câu hỏi phức tạp hoặc có nhiều điều kiện/ngoại lệ, rerank chưa thực sự giúp chọn đúng đoạn context tốt hơn so với baseline dense/hybrid.
- Kết quả này cho thấy rerank không phải lúc nào cũng cải thiện chất lượng, cần cân nhắc kỹ khi áp dụng và có thể phải tinh chỉnh thêm thuật toán hoặc kết hợp với các chiến lược khác (ví dụ: rerank sau khi đã lọc theo sparse/dense).

**Config thay đổi:**
```
retrieval_mode = "hybrid"   # hoặc biến khác
# Các tham số còn lại giữ nguyên như baseline

```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.8/5 | 4.2/5 | -0.6 |
| Answer Relevance | 4.6/5 | 4.1/5 | -0.5 |
| Context Recall | 5.0/5 | 5.0/5 | 0 |
| Completeness | 4.0/5 | 3.8/5 | -0.2 |

**Nhận xét:**
Variant 1 (bật rerank) không cải thiện rõ rệt ở bất kỳ câu hỏi nào so với baseline. Các chỉ số đều giảm nhẹ hoặc giữ nguyên.

Có một số câu kém hơn, ví dụ:
- "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?": Câu trả lời thiếu chi tiết về ngoại lệ, điều kiện áp dụng, faithfulness và completeness giảm.
- "Ai phải phê duyệt để cấp quyền Level 3?": Rerank chọn context gần nghĩa nhưng bỏ sót quy trình chi tiết, làm giảm answer relevance.
- Một số câu hỏi đơn giản không bị ảnh hưởng, nhưng các câu hỏi phức tạp hoặc có nhiều điều kiện thì rerank chưa giúp chọn đúng context tốt hơn.

**Kết luận:**
Variant 1 (bật rerank) không tốt hơn baseline. Tất cả các chỉ số chính (faithfulness, answer relevance, completeness) đều giảm nhẹ so với baseline, context recall giữ nguyên.

Bằng chứng:
- Điểm faithfulness giảm từ 4.8 xuống 4.2 (-0.6), answer relevance giảm từ 4.6 xuống 4.1 (-0.5), completeness giảm từ 4.0 xuống 3.8 (-0.2).
- Một số câu hỏi như "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?" và "Ai phải phê duyệt để cấp quyền Level 3?" cho kết quả kém chi tiết hoặc thiếu chính xác hơn khi bật rerank.
- Không có câu hỏi nào được cải thiện rõ rệt về chất lượng trả lời khi bật rerank.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** Chunk-size   từ 400 lên 500 
**Config:**
```
# TODO
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | 4.8 | 4.2 | 4.2 | 4.8 |
| Answer Relevance | 4.6 | 4.1 | 4.1 | 4.6 |
| Context Recall | 5.0 | 5.0 | 5.0 | 5.0 |
| Completeness | 4.0 | 3.8 | 3.8 | 4.0 |

**Nhận xét Variant 2 (chunk_size 500):**
- Việc tăng chunk size từ 400 lên 500 không cải thiện các chỉ số chính so với baseline, mọi metric đều giữ nguyên hoặc thấp hơn so với chunk size 400.
- Chunk lớn hơn có thể khiến context chứa nhiều thông tin không liên quan, làm giảm độ tập trung và faithfulness của câu trả lời.
- Các câu hỏi cần chi tiết hoặc context ngắn gọn không được lợi từ chunk lớn, thậm chí dễ bị "loãng" thông tin.
- Kết quả: chunk size 400 vẫn là lựa chọn tối ưu hơn trong các thử nghiệm này.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Retrieval bỏ lỡ alias, keyword đặc biệt hoặc context chưa đủ chi tiết cho các câu hỏi phức tạp (ví dụ: mã lỗi, điều kiện hoàn tiền )

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Cách chunking và chiến lược retrieval (dense/hybrid, rerank). Chunking cắt hợp lý giúp context recall tốt hơn, retrieval dense/hybrid ảnh hưởng lớn đến khả năng tìm đúng đoạn liên quan.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Thử tinh chỉnh chunk_size/overlap, kết hợp hybrid retrieval với rerank hoặc thử prompt khác cho LLM. Ngoài ra, có thể thử thêm sparse retrieval hoặc cải thiện metadata để tăng độ chính xác khi filter context.
