#!/usr/bin/env python
# train.py

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

from dataset import CarDamageDataset  # giả sử bạn có class này trong dataset.py

def get_args():
    parser = argparse.ArgumentParser(description="Train Car Damage Segmentation")
    parser.add_argument("--split_dir",    type=str, default="backend/splits",
                        help="folder chứa train.txt, val.txt")
    parser.add_argument("--output_dir",   type=str, default="backend/models",
                        help="nơi lưu checkpoint")
    parser.add_argument("--epochs",       type=int, default=10)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--resize",       type=int, default=512,
                        help="kích thước resize ảnh vuông")
    return parser.parse_args()

def build_transforms(resize):
    return T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
    ])

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)["out"]
            loss = torch.nn.functional.cross_entropy(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = torch.nn.functional.cross_entropy(outputs, masks)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    tf = build_transforms(args.resize)

    # Datasets
    train_ds = CarDamageDataset(
        img_dir=os.path.join(args.split_dir, "train"),
        mask_dir=os.path.join(args.split_dir, "train_masks"),
        transform=tf
    )
    val_ds = CarDamageDataset(
        img_dir=os.path.join(args.split_dir, "val"),
        mask_dir=os.path.join(args.split_dir, "val_masks"),
        transform=tf
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )

    # Model: DeepLabV3-ResNet50
    model = deeplabv3_resnet50(
        pretrained=True, progress=True,
        num_classes=train_ds.num_classes
    )
    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    model.to(device)

    # Optimizer & AMP scaler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scaler = GradScaler()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = validate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        ckpt_path = os.path.join(args.output_dir, "best.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"→ Saved best model to {ckpt_path}")

    # Always save last epoch too
    torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pth"))
    print("Training completed.")

if __name__ == "__main__":
    main()
