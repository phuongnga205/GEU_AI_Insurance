#!/usr/bin/env python
# train.py

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large
)

from dataset import CarDamageDataset  # đảm bảo file dataset.py có class này


def get_args():
    p = argparse.ArgumentParser(description="Train Car Damage Segmentation")
    p.add_argument("--split_dir",      type=str,   default="backend/splits",
                   help="folder chứa train.txt, val.txt, test.txt")
    p.add_argument("--output_dir",     type=str,   default="backend/models",
                   help="nơi lưu checkpoint")
    p.add_argument("--backbone",       type=str,   default="resnet50",
                   choices=["resnet50", "mobilenetv3_large"],
                   help="Chọn backbone cho DeepLabV3")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Nếu được set, freeze toàn bộ backbone")
    p.add_argument("--epochs",         type=int,   default=12)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--resize",         type=int,   default=384,
                   help="Resize ảnh về kích thước vuông")
    p.add_argument("--early_stop",     type=int,   default=5,
                   help="Early stopping sau N epoch không cải thiện")
    return p.parse_args()


def build_transforms(resize):
    return T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
    ])


def load_model(backbone_name, num_classes, freeze_backbone):
    if backbone_name == "mobilenetv3_large":
        model = deeplabv3_mobilenet_v3_large(pretrained=True, num_classes=num_classes)
    else:
        model = deeplabv3_resnet50(pretrained=True, num_classes=num_classes)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
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
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = torch.nn.functional.cross_entropy(outputs, masks)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    tf = build_transforms(args.resize)

    # Datasets (đường dẫn img_dir, mask_dir tương ứng)
    train_ds = CarDamageDataset(
        img_txt=os.path.join(args.split_dir, "train.txt"),
        val=False,
        transform=tf
    )
    val_ds = CarDamageDataset(
        img_txt=os.path.join(args.split_dir, "val.txt"),
        val=True,
        transform=tf
    )
    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model
    model = load_model(args.backbone, train_ds.num_classes, args.freeze_backbone)
    model.to(device)

    # Optimizer & AMP scaler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scaler = GradScaler()

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = validate(model, val_loader, device)
        print(f"  Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping & save best
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            ckpt = os.path.join(args.output_dir, "best.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"→ Saved best model to {ckpt}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print(f"No improvement for {args.early_stop} epochs. Early stopping.")
                break

    # Always save last
    last_ckpt = os.path.join(args.output_dir, "last.pth")
    torch.save(model.state_dict(), last_ckpt)
    print(f"Training completed. Last model at {last_ckpt}")


if __name__ == "__main__":
    main()
