---
title: Elixir như một lớp bộ nhớ
published: 2026-08-02
translationKey: elixir-as-memory-layer
lang: vi
---

- Chèn ngữ cảnh trong lúc chạy nhờ tính đồng thời mà không sợ làm hỏng đầu ra
- Hãy hình dung có một ngữ cảnh bộ nhớ trung tâm trên supervisor / tiến trình chính
- Sau đó, trong lúc chúng ta làm việc, robot sẽ tạo ra các luồng để hiểu hình ảnh / âm thanh / vân vân
- Khi tiến trình trả về phần diễn giải, nó sẽ chèn ngữ cảnh vào bộ nhớ của tiến trình chính
- Vì Elixir vận hành dựa trên tính đồng thời, chúng ta có thể liên tục cập nhật ngữ cảnh bộ nhớ "current" này, đồng thời gửi nó sang các luồng khác để xử lý thành bộ nhớ dài hạn nhằm truy xuất về sau. Chúng ta không cần quan tâm việc đó được thực hiện như thế nào vì đã có BEAM VM.
- Vì mỗi cuộc nói chuyện / trò chuyện tự nó là một tiến trình, chúng ta có thể cứ trò chuyện với agent trong khi nó đang làm nhiệm vụ khác; khả năng này về cơ bản có sẵn mà không cần bất kỳ hệ thống điều phối sub-agent kỳ quặc nào.
- Sau đó, một agent có thể tách ra để trả lời câu hỏi bằng bộ nhớ hiện tại trong một tiến trình riêng, rồi quyết định tạo thêm nhiều tiến trình để gọi tools khi cần nhằm đưa ra câu trả lời.
- Nó cũng có thể làm việc kiểu thông báo rằng nó vẫn đang suy nghĩ cho đến khi tiến trình kia hoàn tất.

