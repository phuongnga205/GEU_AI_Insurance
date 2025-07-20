import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CarDamageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, names_list,
                 img_transform=None, mask_transform=None):
        """
        img_dir:    folder chứa ảnh (.jpg/.jpeg)
        mask_dir:   folder chứa mask (.png)
        names_list: list tên file (không đuôi), vd ["0001", ...]
        img_transform:  torchvision.transforms áp dụng lên ảnh
        mask_transform: torchvision.transforms áp dụng lên mask
        """
        self.img_dir        = img_dir
        self.mask_dir       = mask_dir
        self.img_transform  = img_transform
        self.mask_transform = mask_transform


        # Lọc những sample có cả ảnh và mask
        valid = []
        for name in names_list:
            # check ảnh
            found_img = any(os.path.exists(os.path.join(img_dir, name + ext))
                            for ext in (".jpg", ".jpeg"))
            # check mask
            found_mask = os.path.exists(os.path.join(mask_dir, name + ".png"))
            if found_img and found_mask:
                valid.append(name)
        self.names = valid
        print(f"[Dataset] {len(self.names)}/{len(names_list)} valid samples")


    def __len__(self):
        return len(self.names)


    def __getitem__(self, idx):
        name = self.names[idx]


        # 1) Load image
        for ext in (".jpg", ".jpeg"):
            img_path = os.path.join(self.img_dir, name + ext)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                break


        # 2) Load mask
        mask_path = os.path.join(self.mask_dir, name + ".png")
        mask      = Image.open(mask_path).convert("L")  # grayscale


        # 3) Apply transforms
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # mask tensor will be float in [0,1], convert to binary 0/1
            mask = (mask > 0).float()


        return image, mask
