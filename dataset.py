import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class CarDamageDataset(Dataset):
    """
    Dataset cho segmentation:
      - img_dir: thư mục chứa ảnh .jpg
      - anno_dir: thư mục chứa mask .png (grayscale 0/1)
      - split_file: file .txt liệt kê các tên file (không có đuôi)
      - transform: torchvision.transforms áp cho cả ảnh và mask
    """
    def __init__(self, img_dir, anno_dir, split_file, transform=None):
        self.img_dir   = img_dir
        self.anno_dir  = anno_dir
        self.transform = transform

        # Đọc danh sách tên sample
        with open(split_file, 'r') as f:
            self.names = [line.strip() for line in f if line.strip()]

        # binary segmentation => 1 foreground + 1 background
        self.num_classes = 2

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path  = os.path.join(self.img_dir,  name + '.jpg')
        mask_path = os.path.join(self.anno_dir, name + '.png')

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 0/255

        if self.transform:
            img  = self.transform(img)
            # resize mask, then convert to long tensor of 0/1
            mask = self.transform(mask)
            mask = (mask > 0).long()
        else:
            mask = torch.from_numpy(np.array(mask) > 0).long()

        # mask shape [1,H,W]
        return img, mask
