#!/usr/bin/env python
# train.py

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

from dataset import CarDamageDataset  # class của bạn trong dataset.py


def get_args():
    p = argparse.ArgumentParser(description="Train Car Damage Segmentation")
    p.add_argument("--split_dir",    type=str,   default="backend/splits",
                   help="Folder chứa train.txt, val.txt")
    p.add_argument("--output_dir",   type=str,   default="backend/models",
                   help="Nơi lưu checkpoint")
    p.add_argument("--resume",       type=str,   default=None,
                   help="(Optional) đường dẫn checkpoint .pth để resume training")
    p.add_argument("--epochs",       type=int,   default=20,
                   help="Số epoch tối đa")
    p.add_argument("--batch_size",   type=int,   default=8,
                   help="Batch size")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--workers",      type=int,   default=4,
                   help="Số workers cho DataLoader")
    p.add_argument("--resize",       type=int,   default=512,
                   help="Kích thước resize ảnh (vuông)")
    p.add_argument("--early_stop",   type=int,   default=5,
                   help="Dừng sớm sau N epoch không cải thiện val loss")
    return p.parse_args()


def build_transforms(resize):
    return T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
    ])


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = torch.nn.functional.cross_entropy(outputs, masks)
        loss.backward()
        optimizer.step()
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
    print(f"Using device: {device}")

    # Transforms
    transforms = build_transforms(args.resize)

    # Datasets & DataLoaders
    train_ds = CarDamageDataset(
        img_txt=os.path.join(args.split_dir, "train.txt"),
        val=False, transform=transforms
    )
    val_ds = CarDamageDataset(
        img_txt=os.path.join(args.split_dir, "val.txt"),
        val=True, transform=transforms
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model: DeepLabV3 + ResNet101, binary segmentation
    model = deeplabv3_resnet101(pretrained=True, num_classes=1)
    model.to(device)

    # Resume from checkpoint nếu có
    if args.resume:
        print(f"→ Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"  Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Nếu val_loss cải thiện thì lưu checkpoint
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(args.output_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"→ Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print(f"No improvement for {args.early_stop} epochs, stopping.")
                break

    # Luôn lưu checkpoint cuối cùng
    last_path = os.path.join(args.output_dir, "last.pth")
    torch.save(model.state_dict(), last_path)
    print(f"Training completed. Last model saved to {last_path}")


if __name__ == "__main__":
    main()
