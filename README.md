# GEU\_AI\_Insurance 🚗🔧

## 👩‍💼 Thông tin tác giả

- **Trường**: Đại học Công nghệ Thông tin – ĐHQG TP.HCM
- **MSSV**: 23520992
- **Họ tên**: Lê Ngọc Phương Nga

## 🌟 Tổng quan dự án

GEU\_AI\_Insurance là một hệ thống trí tuệ nhân tạo ứng dụng học sâu (Deep Learning) để phát hiện và phân vùng (segmentation) vùng bị hư hại trên xe ô tô từ hình ảnh. Ngoài ra, hệ thống còn có chức năng ước tính tổn thất (damage cost estimation) và cung cấp nền tảng để triển khai một giao diện web giúp người dùng tải ảnh và xem kết quả dự đoán trực quan.

## 🔍 Các chức năng chính

- Huấn luyện mô hình nhận diện và phân vùng tổn thất từ ảnh xe.
- Hiển thị trực quan vùng hư hại dự đoán.
- Hỗ trợ ước tính tổn thất dựa trên tỷ lệ vùng hư hại.
- Cung cấp giao diện web đơn giản để người dùng tải ảnh và nhận kết quả (đang triển khai).

## ⚙️ Công nghệ sử dụng

### 🧠 Backend - AI & Deep Learning

- **Python 3.8+**
- **PyTorch**: Framework chính để xây dựng và huấn luyện mô hình segmentation.
  - Lý do chọn: Dễ tùy biến và phổ biến trong cộng đồng ML.
- **OpenCV**: Tiền xử lý và hiển thị hình ảnh.
- **NumPy**: Xử lý dữ liệu hiệu quả.
- **Matplotlib**: Trực quan hóa kết quả segmentation.

### 🌐 Frontend - Giao diện web (sẽ triển khai)

- **HTML/CSS** (Jinja2 templates): Tạo giao diện đơn giản cho người dùng tải ảnh.
- **Flask**: Web framework lightweight, dễ tích hợp mô hình ML vào API backend.
  - Lý do chọn: Nhẹ, dễ triển khai, nhanh chóng xây dựng MVP (Minimum Viable Product).

## 📁 Cấu trúc thư mục dự án

```plaintext
GEU_AI_Insurance/
├─ app/
│  └─ web/                     # Code web demo
│     ├─ __init__.py           # Python package
│     ├─ main.py               # Flask app
│     ├─ utils.py              # Load model & process image
│     ├─ templates/
│     │  └─ index.html         # Giao diện upload & kết quả
│     └─ static/
│         ├─ uploads/          # Ảnh gốc của người dùng
│         └─ results/          # Ảnh đã overlay
├─ sample_car_damage/          # Dữ liệu demo ảnh xe
├─ annotations/                # File chú thích LabelMe
├─ source_code/
│  ├─ backend/                 # Mã nguồn ML và training
│  └─ frontend/                # Mã nguồn giao diện web (Flask)
├─ models/checkpoints/         # Lưu model .pth đã huấn luyện
├─ screenshots/                # Hình ảnh demo sản phẩm
├─ train.py                    # Script huấn luyện model
├─ requirements.txt            # Dependencies
├─ README.md                   # Hướng dẫn sử dụng & mô tả dự án
└─ TRAINING.md                 # Mô tả pipeline và metrics
```

## 📞 Liên hệ

Nếu bạn cần hỗ trợ hoặc muốn đóng góp, vui lòng liên hệ qua:

- GitHub: [https://github.com/phuongnga205/GEU\_AI\_Insurance](https://github.com/phuongnga205/GEU_AI_Insurance)
- Email: [23520992@gm.uit.edu.vn](mailto:23520992@gm.uit.edu.vn)

