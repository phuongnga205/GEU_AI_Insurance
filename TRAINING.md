
# 🏋️‍♂️ TRAINING.md — Huấn luyện mô hình nhận diện hư hỏng xe ô tô

## 1. 🎯 Mục tiêu

Huấn luyện mô hình segmentation để nhận diện chính xác vùng hư hỏng trên thân xe ô tô từ ảnh đầu vào.

---

## 2. 🧹 Xử lý dữ liệu (Data Processing)

- Dữ liệu gốc gồm 2 thư mục:
  - `app/sample_car_damage/`: chứa ảnh gốc xe (JPG/PNG).
  - `app/annotations/`: chứa ảnh mask phân vùng vùng hư hỏng (cùng tên file, định dạng PNG/gray).

- Bước xử lý:
  1. Resize ảnh về kích thước chuẩn: `256×256` (hoặc `512×512` tùy GPU).
  2. Chuẩn hóa ảnh (Normalize theo mean/std ImageNet hoặc [0,1]).
  3. Áp dụng Augmentation cơ bản:
     - Horizontal flip (ngẫu nhiên)
     - Rotation ±10°
     - Brightness/Contrast shift
  4. Chia tập train/val/test bằng `make_splits.py` theo tỉ lệ:
     - Train: 70%
     - Validation: 15%
     - Test: 15%
  5. Tạo custom Dataset class bằng PyTorch: `CarDamageDataset` (file dataset.py).

---

## 3. 🧠 Mô hình và Thuật toán

- 📌 Mô hình: U-Net (UNet encoder-decoder segmentation)
  - Encoder: ResNet34 pretrained (ImageNet)
  - Decoder: Upsample + Skip connections
- 📦 Framework: PyTorch
- 💻 File huấn luyện: `train.py`

- Hàm Loss:
  - `DiceLoss` kết hợp với `Binary Cross Entropy (BCE)`
    ```python
    loss = 0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss
    ```
- Tối ưu:
  - Optimizer: Adam
  - Learning rate: 1e-4
  - Scheduler: ReduceLROnPlateau
  - Epochs: 2
  - Batch size: 8 (tuỳ cấu hình GPU)

---

## 6. 💾 Xuất mô hình

- Model được lưu lại bằng:
  ```python
  torch.save(model.state_dict(), 'models/checkpoints/unet_best.pth')
  ```
- Được lưu trên link https://drive.google.com/drive/folders/19Ic_gaaYdSDZoAg59JTNoCQVPqU2ANvQ 
