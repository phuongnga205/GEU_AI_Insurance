import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
import torchvision.transforms as T
from tqdm import tqdm

from dataset import CarDamageDataset  # lớp Dataset bạn đã tự viết

def get_transforms():
    # Transform cho ảnh: resize, to_tensor, normalize
    img_tf = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # Transform cho mask: resize nearest, to_tensor
    mask_tf = T.Compose([
        T.Resize((520, 520), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),   # sẽ ra float [0,1]
    ])
    return img_tf, mask_tf

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)      # [B,3,520,520]
        masks  = masks.to(device)       # [B,1,520,520]

        optimizer.zero_grad()
        outputs = model(images)['out']  # [B,1,520,520]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  Validating", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # 1) Đọc danh sách tên file từ train.txt, val.txt
    with open("train.txt") as f:
        train_names = [l.strip() for l in f if l.strip()]
    with open("val.txt") as f:
        val_names   = [l.strip() for l in f if l.strip()]

    # 2) Transforms
    img_tf, mask_tf = get_transforms()

    # 3) Khởi tạo dataset
    train_ds = CarDamageDataset(
        img_dir="sample_car_damage",
        mask_dir="annotations",
        names_list=train_names,
        img_transform=img_tf,
        mask_transform=mask_tf
    )
    val_ds = CarDamageDataset(
        img_dir="sample_car_damage",
        mask_dir="annotations",
        names_list=val_names,
        img_transform=img_tf,
        mask_transform=mask_tf
    )

    # 4) DataLoader: thêm drop_last=True để bỏ batch cuối thiếu mẫu
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=True      # ← đây
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False     # val có thể giữ False
    )

    # 5) Model, criterion, optimizer
    model = deeplabv3_resnet101(pretrained=True)
    # thay head classifier thành 1 channel (damage vs background)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 6) Training loop
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, 11):
        print(f"\nEpoch {epoch}/10")
        tr_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"  Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Lưu model nếu val_loss giảm
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = f"checkpoints/best_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best model to {ckpt_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
