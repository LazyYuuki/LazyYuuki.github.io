---
title: Dùng công nghệ nào khi nào?
published: 2026-08-02
translationKey: which-tech-when
lang: vi
---

Những câu hỏi cần đặt ra khi bắt đầu một dự án:
- Nó cần phục vụ rất nhiều người dùng hay chỉ một vài người dùng đơn lẻ?
- Nó cần duy trì nhiều kết nối tồn tại lâu không?
- Nó cần xử lý nhiều yêu cầu mỗi giây không?
- Nó có cần khả năng chịu lỗi và mở rộng cao không?
- Nó tối ưu cho tính đồng thời hay tốc độ?

### Theo tôi:

#### Zig, Rust, C, Go 
- Hệ thống nhúng
- Ứng dụng nhỏ, nhanh dành cho một người dùng
- Tiến trình đáng tin cậy
- Rất nhiều yêu cầu HTTP mỗi giây
- TỐỐỐỐC ĐỘỘỘỘ

#### Elixir
- Tính đồng thời ở quy mô lớn
- Hệ thống trò chuyện
- Những thứ liên quan đến thời gian thực như dashboard, messaging và ứng dụng cộng tác

#### Python
- Chỉ để thử nghiệm và thăm dò
- Không bao giờ nên xây dựng hệ thống thực tế bằng nó

