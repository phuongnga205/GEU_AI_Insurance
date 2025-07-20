import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class CarDamageDataset(Dataset):
    """
    Dataset cho segment car damage.
    - img_dir: thư mục chứa ảnh .jpg
    - anno_dir: thư mục chứa mask .png (hoặc .bmp) cùng tên file
    - split_file: đường dẫn tới file .txt liệt kê tên (không có đuôi) các sample
    - transform: torchvision.transforms áp cho ảnh (không áp mask!)
    """
    def __init__(self, img_dir, anno_dir, split_file, transform=None):
        self.img_dir   = img_dir
        self.anno_dir  = anno_dir
        self.transform = transform

        # Đọc danh sách tên sample (ví dụ dòng "000123")
        with open(split_file, 'r') as f:
            self.names = [line.strip() for line in f if line.strip()]

        # Nếu mask là binary (0/1), num_classes=2; nếu chỉ 1 lớp foreground, num_classes=1
        self.num_classes = 2

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        # load ảnh
        img_path  = os.path.join(self.img_dir,  name + '.jpg')
        mask_path = os.path.join(self.anno_dir, name + '.png')
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale mask

        # áp transform cho ảnh (không áp trực tiếp mask, chỉ resize v.v.)
        if self.transform:
            img = self.transform(img)
            # mask resize phải dùng nearest để giữ nhãn nguyên
            mask = self.transform(mask).long()

        # mask: tensor kích thước [1,H,W] với giá trị 0 hoặc 1
        return img, mask

