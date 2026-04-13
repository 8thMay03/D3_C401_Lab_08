# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Lưu Thị Ngọc Quỳnh
**Vai trò trong nhóm:** Documentation Owner  
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500 - 800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi chủ yếu đảm nhận phần tài liệu, nhưng không chỉ ghi chép kết quả mà còn phải hiểu được logic của pipeline để viết cho đúng. Các phần tôi tham gia cụ thể gồm:

- Tôi phụ trách vai trò Documentation Owner, tập trung nhiều nhất ở Sprint 4 để hoàn thiện các tài liệu tổng kết của nhóm.
- Tôi cập nhật `docs/architecture.md` dựa trên kết quả chạy thực tế, bao gồm kiến trúc pipeline, cấu hình indexing, retrieval, generation và evaluation.
- Tôi chuẩn bị nội dung cho `docs/tuning-log.md`, tổng hợp kết quả baseline và variant để nhóm có thể giải thích phần A/B comparison rõ ràng hơn.
- Tôi đọc và theo sát `rag_answer.py`, đặc biệt là các hàm `rerank()`, `build_context_block()`, `build_grounded_prompt()`, `call_llm()`, `transform_query()` và `rag_answer()`.
- Tôi kết nối phần tài liệu với phần code của các bạn khác bằng cách chuyển các kết quả kỹ thuật thành nội dung dễ hiểu, có thể dùng để báo cáo và demo.

Nhờ làm phần này, tôi thấy rõ tài liệu không phải phần “viết lại cho đẹp” mà là nơi tổng hợp và giải thích quyết định kỹ thuật của cả nhóm.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau khi làm lab, tôi hiểu rõ hơn một số khái niệm mà trước đây tôi chỉ biết ở mức lý thuyết. Hai điểm tôi thấy mình tiến bộ rõ nhất là:

- Tôi hiểu rõ hơn về `rerank()`: retrieve đúng tài liệu chưa đủ, vì trong top-k vẫn có thể có nhiều chunk nhiễu. Rerank dùng cross-encoder để chấm lại mức liên quan giữa query và từng chunk, từ đó chọn ra các đoạn phù hợp nhất.
- Tôi hiểu grounded prompt không đơn thuần là prompt “trả lời ngắn”, mà là một cơ chế ràng buộc mô hình chỉ được trả lời từ evidence đã retrieve.
- Tôi thấy `build_context_block()` rất quan trọng vì cách sắp xếp source, section và score ảnh hưởng trực tiếp đến việc model có bám đúng tài liệu hay không.
- Tôi cũng hiểu hơn vai trò của `build_grounded_prompt()` trong việc yêu cầu model abstain khi thiếu dữ liệu, thay vì tự bổ sung kiến thức ngoài ngữ cảnh.

Qua đó, tôi rút ra rằng chất lượng của hệ thống RAG không chỉ nằm ở model sinh câu trả lời, mà nằm ở toàn bộ cách chuẩn bị context trước khi gọi model.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Trong quá trình làm, có vài điểm khiến tôi khá bất ngờ và cũng mất thời gian để hiểu rõ:

- Tôi ngạc nhiên vì variant không tốt hơn baseline như kỳ vọng ban đầu, dù variant có thêm hybrid retrieval và rerank.
- Trước khi xem scorecard, tôi nghĩ thêm nhiều kỹ thuật hơn sẽ làm chất lượng tăng lên rõ rệt. Thực tế, baseline dense lại có điểm trung bình tốt hơn ở một số metric quan trọng.
- Khó khăn lớn nhất là phân biệt lỗi nằm ở indexing, retrieval hay generation, vì nhiều khi câu trả lời sai nhưng retrieve lại đúng nguồn.
- Một lỗi kỹ thuật đáng chú ý là `ChromaDB` báo mismatch embedding dimension `384` và `1536`, cho thấy nếu index bằng một embedding model rồi chuyển sang model khác để query thì collection cũ sẽ không dùng tiếp được.
- Ở góc độ viết tài liệu, phần khó là diễn giải nguyên nhân kỹ thuật theo cách dễ hiểu, thay vì chỉ chép lại log hoặc mô tả hiện tượng bề mặt.

Sau phần này, tôi thấy việc đọc kết quả evaluation quan trọng không kém việc chạy pipeline, vì nếu không phân tích kỹ thì rất dễ kết luận sai nguyên nhân.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q09 - ERR-403-AUTH là lỗi gì và cách xử lý?`

**Phân tích:**

Tôi chọn câu hỏi này vì nó cho thấy rất rõ một trường hợp khó trong RAG: hệ thống không có dữ liệu nhưng vẫn phải trả lời an toàn và hữu ích. Khi xem scorecard, tôi phân tích câu này theo các ý sau:

- Đây là câu hỏi về mã lỗi `ERR-403-AUTH`, nhưng trong bộ tài liệu hiện có không có chunk nào mô tả trực tiếp lỗi này.
- Ở baseline, câu trả lời được chấm faithfulness `5`, relevance `5`, completeness `3`. Điều đó cho thấy baseline xử lý khá an toàn, không bịa nội dung ngoài tài liệu.
- Tuy nhiên, baseline vẫn chưa đạt completeness cao vì câu trả lời chưa đưa thêm hướng dẫn hỗ trợ như nên liên hệ bộ phận nào hoặc làm gì tiếp theo.
- Ở variant hybrid + rerank, điểm số lại giảm mạnh xuống faithfulness `1`, relevance `1`, completeness `1`, dù về bản chất hệ thống vẫn đang gặp cùng một tình huống thiếu dữ liệu.
- Theo tôi, lỗi chính ở đây không nằm ở retrieval, vì corpus thật sự không có thông tin cần tìm. Vấn đề nằm nhiều hơn ở generation và tiêu chí chấm điểm cho câu trả lời dạng abstain.
- Câu “Tôi không biết” tuy an toàn nhưng quá ngắn, nên bị judge xem là thiếu hỗ trợ cho người dùng.

Từ câu này, tôi học được rằng trong RAG, trả lời “không biết” chưa chắc đã đủ. Nếu có thêm thời gian, tôi sẽ đề xuất chuẩn hóa mẫu abstain theo hướng an toàn nhưng vẫn hữu ích, ví dụ: “Không tìm thấy thông tin trong tài liệu hiện có, vui lòng liên hệ IT Helpdesk để được hỗ trợ thêm.”

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi muốn tiếp tục cải thiện cả phần tài liệu lẫn cách nhóm đánh giá pipeline:

- Tôi sẽ đề xuất chuẩn hóa mẫu câu trả lời khi thiếu dữ liệu để các câu abstain không bị quá ngắn và mất điểm relevance/completeness.
- Tôi muốn nhóm thử tách riêng từng biến trong Sprint 3, ví dụ chỉ bật `rerank` hoặc chỉ bật `hybrid`, thay vì kết hợp cùng lúc.
- Tôi cũng muốn cập nhật `tuning-log.md` theo hướng chi tiết hơn, để mỗi thay đổi đều có lý do, kết quả và kết luận rõ ràng.

Nếu làm được các bước này, nhóm sẽ có tài liệu chặt chẽ hơn và cũng dễ giải thích hơn vì sao một variant thực sự hiệu quả hoặc không hiệu quả.

---
