# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Quốc Khánh  
**Vai trò trong nhóm:** Tech Lead / Retrieval Owner
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này tôi tập trung nhiều nhất ở Sprint 2 và Sprint 3 với vai trò Retrieval Owner, đồng thời hỗ trợ điều phối kỹ thuật ở mức Tech Lead. Phần chính tôi làm là xây dựng luong truy xuất tài liệu cho pipeline RAG: chuẩn hóa chunk metadata, chọn tham số top-k, và thử nghiệm kết hợp dense retrieval với BM25 để tạo hybrid candidate set. Tôi trực tiếp chỉnh logic ở `rag_answer.py` cho các bước `retrieve_candidates()`, `rerank()`, và `build_context_block()` để context đưa vào model vừa đủ ngắn nhưng vẫn giữ thông tin cốt lõi. Ngoài ra tôi phối hợp với bạn phụ trách generation để thống nhất grounded prompt theo định dạng evidence rõ source, giúp model bám vào tài liệu thay vì suy diễn. Kết quả retrieval và score từng variant được tôi bàn giao cho bạn Documentation Owner để tổng hợp vào tuning log, từ đó cả nhóm có cùng dữ liệu khi so sánh baseline và variant.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này tôi hiểu rõ hơn hai điểm: hybrid retrieval và evaluation loop. Trước đây tôi nghĩ chỉ cần embedding model tốt là dense retrieval sẽ luôn đủ mạnh. Khi chạy thực tế, tôi thấy dense thường bỏ sót các câu hỏi chứa keyword đặc thù (mã lỗi, tên policy, viết tắt). Hybrid retrieval giải quyết điểm mù đó bằng cách lấy candidate từ cả semantic similarity lẫn lexical match, sau đó mới rerank để chọn đoạn liên quan nhất. Tôi cũng hiểu evaluation loop không phải bước "chấm điểm cuối cùng" mà là cơ chế phản hồi để chỉnh ngược lại retrieval/generation. Ví dụ khi điểm faithfulness thấp, chưa chắc lỗi ở model sinh; có thể context vào đã nhiễu do top-k quá lớn hoặc rerank chưa hiệu quả. Nhờ cách nhìn theo vòng lặp này, tôi chuyển từ tối ưu cảm tính sang tối ưu dựa trên evidence và metric cụ thể.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi bất ngờ nhất là có thời điểm variant "phức tạp hơn" lại cho điểm kém baseline. Giả thuyết ban đầu của tôi là thêm hybrid retrieval và rerank thì chất lượng phải tăng đồng đều. Thực tế cho thấy khi candidate pool quá rộng, một số chunk nhiễu vẫn lọt vào context, làm model trả lời lan man dù retrieve tưởng như "đầy đủ". Lỗi mất thời gian debug nhất là vấn đề embedding dimension không khớp khi thay model trong lúc tái sử dụng collection cũ; hệ thống báo mismatch khiến query chạy không ổn định giữa các lần thử. Từ lỗi này tôi rút kinh nghiệm phải cố định cấu hình embedding theo từng index version và ghi rõ trong pipeline log. Khó khăn thứ hai là phân tách nguyên nhân sai số theo tầng indexing/retrieval/generation; nếu chỉ nhìn output cuối rất dễ kết luận nhầm và chỉnh sai chỗ.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q09 - ERR-403-AUTH là lỗi gì và cách xử lý?`

**Phân tích:**

Tôi chọn câu này vì nó kiểm tra khả năng "không bịa" khi corpus không có thông tin trực tiếp. Ở baseline, điểm của câu `q09` là Faithfulness 5, Relevance 5, Completeness 3; hệ thống trả lời theo hướng thiếu dữ liệu nên tránh được hallucination, nhưng vẫn thiếu phần hướng dẫn hành động rõ ràng. Khi chuyển sang variant hybrid + rerank, điểm giảm mạnh còn Faithfulness 1, Relevance 1, Completeness 1. Nguyên nhân là pipeline lấy thêm nhiều candidate chứa từ khóa gần giống "auth" nhưng không nói đúng về `ERR-403-AUTH`, khiến ngữ cảnh đầu vào bị loãng. Khi đó model hoặc trả lời quá ngắn kiểu abstain, hoặc bám nhầm vào policy liên quan xác thực nhưng không đúng mã lỗi được hỏi.

Theo tôi, lỗi chính nằm ở retrieval-to-context selection hơn là indexing. Index vẫn hoạt động, nhưng chiến lược chọn final context chưa đủ chặt khi không có evidence thật sự trùng đích. Generation chỉ bộc lộ hậu quả của context nhiễu. Bài học rút ra là với câu hỏi dạng mã lỗi hiếm, cần thêm cơ chế confidence gate: nếu không có đoạn nào đạt ngưỡng liên quan tối thiểu thì trả lời abstain có cấu trúc (nêu rõ "không có trong tài liệu" + hướng dẫn liên hệ hỗ trợ), thay vì cố trả lời theo các chunk tương tự bề mặt.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ thử hai cải tiến cụ thể. Thứ nhất, thêm ngưỡng confidence sau rerank để chặn các context "gần đúng" nhưng không đúng trọng tâm, vì eval cho thấy một số câu giảm điểm do nhiễu ngữ cảnh. Thứ hai, tôi muốn tách ablation rõ ràng (baseline -> +hybrid -> +rerank -> +prompt tuning) để biết chính xác bước nào tạo lợi ích thực, thay vì bật nhiều thay đổi cùng lúc rồi khó quy nguyên nhân.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*