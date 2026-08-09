---
title: Suy nghĩ về Cline [Đã lỗi thời]
description: Một vài suy nghĩ về việc sử dụng Cline
published: 2025-02-06T09:00:00+08:00
draft: false
lang: vi
translationKey: thought-on-cline
---

Tôi tò mò về cách Cline hoạt động và liệu công nghệ dành cho LLM thực hiện các tác vụ có context dài đã thay đổi đáng kể trong vài năm qua hay chưa.

Và tôi đi đến một kết luận đơn giản rằng chẳng có gì thay đổi nhiều!

Cline vẫn hoạt động đúng theo cách chúng ta tạo khả năng tương tác cho LLM ngay từ thuở ban đầu: nhồi mọi thứ vào context, bao gồm cả một lượng rất lớn dữ liệu context lịch sử.

Khác biệt chính là giờ đây mô hình có context window lớn hơn nhiều, chi phí cho mỗi triệu token rẻ hơn nhiều và tốc độ inference nhanh hơn nhiều. Nhưng về cơ bản, khái niệm context stuffing vẫn không thay đổi.

Điều này có nghĩa là khi bạn làm việc với codebase và dự án ngày càng lớn, sẽ đến lúc chỉ một query thôi cũng đắt đến vô lý, khiến việc nhờ LLM thực hiện tác vụ không còn hợp lý nữa. Hoặc đơn giản là do bản chất hữu hạn của context window, nó sẽ không thể tiếp nhận thêm nội dung.

Dù vậy, đã có khá nhiều cải tiến nhằm giảm chi phí sử dụng LLM. Một ví dụ tiêu biểu là sự ra đời của prompt caching. Do đặc thù làm việc với LLM trong một codebase, khi cố gắng hoàn thành 1 tác vụ, bạn thường trao đổi liên tục với mô hình LLM cho đến khi tác vụ đó hoàn tất. Vậy thay vì luôn gửi một prompt mới kèm theo toàn bộ prompt cũ dưới dạng một request mới, tại sao chúng ta không cache lịch sử trong LLM và giữ cache đó ở trạng thái warm trong suốt thời gian còn sử dụng?

Prompt caching hoạt động bằng cách giữ cache của tất cả message trước đó. Nếu đã xử lý một request, nó sẽ giữ lại phần prompt hoàn toàn giống nhau và dùng phần đã được tính toán trước đó khớp với prompt mới để đưa vào inference, thay vì phải tính toán lại trên cùng một text prompt.

Nhược điểm duy nhất là prompt cache chỉ duy trì trạng thái warm trong 5 đến 10 phút. Vì vậy, nếu bạn để một phiên tác vụ mở quá lâu trên Cline, toàn bộ context length sẽ lại được gửi đến LLM để xử lý, và việc đó sẽ không hề rẻ.

Ngoài điều đó ra, công nghệ vẫn như cũ và vẫn bị giới hạn bởi context length của mô hình bạn đang sử dụng.

Dưới đây là một số dữ liệu ví dụ cho thấy cách nó hoạt động:

Mỗi request luôn "sent" (lần đầu từ prompt của chúng ta, lần thứ hai từ cache) toàn bộ system prompt. Đó là lý do trong 2 request, số token được gửi là 28.7k trong khi context window chỉ khoảng 14k.

Thứ hai, bạn có thể nhanh chóng nhận ra rằng dù gửi chính xác cùng một system prompt, request đầu tiên tốn $0.0570 còn request thứ hai tốn $0.0101. Điều này có được nhờ prompt caching trên dịch vụ Anthropic, nơi chỉ tính phí bằng 10% mức giá khi đọc prompt từ cache.

Đôi khi sử dụng Cline, bạn cũng sẽ nhận thấy request đầu tiên của một tác vụ mới rất rẻ:

Đó là vì cache từ các tác vụ trước của bạn vẫn còn warm, nên phần system prompt được giữ lại và giúp bạn không phải xử lý lại toàn bộ prompt.

Khá thú vị là tôi hoàn toàn không thể tìm thấy bất kỳ thông tin nào về cách cơ chế này hoạt động bên trong.

Một điều khác cần lưu ý là bạn càng thêm nhiều MCP vào cline thì càng phải trả nhiều tiền cho mỗi request, ngay cả khi không sử dụng khả năng của các công cụ đó. Tôi nghĩ đây là điều mọi người dường như thường bỏ qua. Nó không giống như tải một plugin hay extension về IDE hoặc ứng dụng của bạn rồi xong.

Không. Nó giống một gói thuê bao hơn. Bạn sẽ tiếp tục trả tiền mỗi khi gửi prompt và trong suốt thời gian nó còn nằm trong danh sách MCP servers của mình. Vì vậy, hãy chọn lọc những gì bạn muốn thêm vào danh sách MCP servers và đừng tham lam.
