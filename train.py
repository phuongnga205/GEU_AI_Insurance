#!/usr/bin/env python
# train.py

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

from dataset import CarDamageDataset  # dataset.py không đổi


def get_args():
    p = argparse.ArgumentParser("Train Car Damage Segmentation")
    p.add_argument("--split_dir",  type=str, default="backend/splits",
                   help="Thư mục chứa train.txt, val.txt, test.txt")
    p.add_argument("--output_dir", type=str, default="backend/models",
                   help="Nơi lưu các checkpoint")
    p.add_argument("--epochs",     type=int,   default=20,
                   help="Số epoch tối đa")
    p.add_argument("--batch_size", type=int,   default=8,
                   help="Batch size")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--workers",    type=int,   default=4,
                   help="Số workers cho DataLoader")
    p.add_argument("--resize",     type=int,   default=512,
                   help="Kích thước resize ảnh (vuông)")
    p.add_argument("--early_stop", type=int,   default=5,
                   help="Early stop sau N epoch không cải thiện")
    return p.parse_args()


def build_transforms(resize):
    return T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
    ])


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs  = imgs.to(device)
        masks = masks.squeeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(imgs)["out"]
        loss = torch.nn.functional.cross_entropy(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs  = imgs.to(device)
            masks = masks.squeeze(1).to(device)
            outputs = model(imgs)["out"]
            loss = torch.nn.functional.cross_entropy(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Transforms
    tf = build_transforms(args.resize)

    # 2) Load split files bằng args.split_dir
    train_txt = os.path.join(args.split_dir, "train.txt")
    val_txt   = os.path.join(args.split_dir, "val.txt")

    # 3) Dataset
    train_ds = CarDamageDataset(
        img_dir    = "sample_car_damage",
        anno_dir   = "annotations",
        split_file = train_txt,
        transform  = tf
    )
    val_ds = CarDamageDataset(
        img_dir    = "sample_car_damage",
        anno_dir   = "annotations",
        split_file = val_txt,
        transform  = tf
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 4) Model
    model = deeplabv3_resnet101(pretrained=True, num_classes=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val   = float("inf")
    no_improve = 0

    # 5) Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        v_loss  = validate(model, val_loader, device)
        print(f" Train loss: {tr_loss:.4f} | Val loss: {v_loss:.4f}")

        if v_loss < best_val:
            best_val   = v_loss
            no_improve = 0
            ckpt_path  = os.path.join(args.output_dir, "best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(" Saved best model:", ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping after {no_improve} epochs without improvement")
                break

    # 6) Save last checkpoint
    last_path = os.path.join(args.output_dir, "last.pth")
    torch.save(model.state_dict(), last_path)
    print("Training complete. Last model saved to", last_path)


if __name__ == "__main__":
    main()
