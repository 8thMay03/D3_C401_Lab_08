# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Phương Nam
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi chủ yếu tham gia vào giai đoạn xây dựng và tối ưu phần retrieval trong RAG pipeline, đặc biệt ở sprint 2 và sprint 3. Cụ thể, tôi chịu trách nhiệm thiết kế cách chia nhỏ dữ liệu (chunking) và cấu hình truy vấn để lấy ra top-k chunk phù hợp nhất từ vector store. Tôi cũng thử nghiệm các giá trị khác nhau của k để cân bằng giữa độ chính xác và độ nhiễu của context. Ngoài ra, tôi phối hợp với thành viên phụ trách generation để đảm bảo các chunk được retrieve có thể được đưa vào prompt một cách rõ ràng và có cấu trúc. Công việc của tôi đóng vai trò trung gian, kết nối giữa indexing (chuẩn bị dữ liệu) và generation (tạo câu trả lời).
_________________

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn về cách chunking ảnh hưởng trực tiếp đến chất lượng retrieval. Trước đây tôi nghĩ chỉ cần chia văn bản thành các đoạn nhỏ là đủ, nhưng thực tế việc chia quá nhỏ sẽ làm mất ngữ cảnh, còn chia quá lớn lại khiến embedding kém chính xác. Ngoài ra, tôi cũng hiểu rõ hơn về khái niệm “top-k retrieval”. Không phải lúc nào lấy nhiều chunk hơn cũng tốt, vì nếu k quá lớn, model sẽ bị nhiễu thông tin và khó tập trung vào nội dung quan trọng. Việc chọn k phù hợp là một trade-off giữa recall và precision, và cần được điều chỉnh dựa trên kết quả evaluation thực tế.

_________________

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Một điều khiến tôi khá bất ngờ là việc retrieval có thể trả về các chunk có vẻ “liên quan” về mặt từ khóa nhưng lại không thực sự giúp ích cho việc trả lời câu hỏi. Ban đầu tôi nghĩ embedding sẽ giúp giải quyết hoàn toàn vấn đề này, nhưng thực tế vẫn có nhiều trường hợp retrieve sai ngữ cảnh. Khó khăn lớn nhất là debug khi kết quả cuối cùng sai: rất khó xác định lỗi đến từ retrieval hay generation. Giả thuyết ban đầu của tôi là lỗi nằm ở model sinh câu trả lời, nhưng sau khi kiểm tra kỹ thì nhận ra nhiều trường hợp chunk được retrieve đã không chứa đủ thông tin cần thiết. Điều này khiến tôi phải kiểm tra lại toàn bộ pipeline thay vì chỉ tập trung vào một bước.
_________________

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** What is the main purpose of the RAG pipeline?

**Phân tích:**
Ở baseline, hệ thống trả lời chưa chính xác hoàn toàn. Mặc dù câu trả lời có đề cập đến việc kết hợp retrieval và generation, nhưng lại thiếu phần giải thích rõ ràng về vai trò của việc đưa context từ bên ngoài vào để tăng độ chính xác. Điểm số vì vậy chỉ ở mức trung bình. Sau khi kiểm tra, nhóm xác định lỗi chủ yếu nằm ở bước retrieval: các chunk được lấy về không chứa định nghĩa đầy đủ về RAG mà chỉ là các đoạn mô tả rời rạc.

Ở variant sau khi cải thiện chunking và giảm giá trị k, kết quả đã tốt hơn. Các chunk được retrieve tập trung hơn vào phần định nghĩa, giúp model có đủ thông tin để trả lời đầy đủ và chính xác hơn. Điều này cho thấy việc tối ưu retrieval có tác động rất lớn đến chất lượng đầu ra, thậm chí còn quan trọng hơn việc thay đổi prompt ở bước generation.
_________________

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi muốn thử áp dụng hybrid retrieval (kết hợp semantic search và keyword search) để cải thiện độ chính xác trong các câu hỏi yêu cầu thông tin cụ thể. Ngoài ra, tôi cũng muốn xây dựng một evaluation loop tự động để test nhiều giá trị k khác nhau và chọn ra cấu hình tối ưu dựa trên score thực tế, thay vì điều chỉnh thủ công như hiện tại.
_________________

---
