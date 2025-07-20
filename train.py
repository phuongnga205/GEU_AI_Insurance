#!/usr/bin/env python
import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

from dataset import CarDamageDataset  # class cũ nhận img_txt, val, transform


def get_args():
    p = argparse.ArgumentParser("Train CarDamageSeg (original)")
    p.add_argument("--split_dir",  type=str, default="backend/splits",
                   help="Folder chứa train.txt, val.txt")
    p.add_argument("--output_dir", type=str, default="backend/models",
                   help="Nơi lưu checkpoint")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--resize",     type=int,   default=512)
    p.add_argument("--early_stop", type=int,   default=5)
    return p.parse_args()


def build_transforms(resize):
    return T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
    ])


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.squeeze(1).to(device)
        opt.zero_grad()
        out = model(imgs)["out"]
        loss = torch.nn.functional.cross_entropy(out, masks)
        loss.backward()
        opt.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(device), masks.squeeze(1).to(device)
            out = model(imgs)["out"]
            loss = torch.nn.functional.cross_entropy(out, masks)
            total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tf = build_transforms(args.resize)

    # **Dùng signature cũ của CarDamageDataset(img_txt, val, transform)**
    train_ds = CarDamageDataset(
        img_txt = os.path.join(args.split_dir, "train.txt"),
        val     = False,
        transform = tf
    )
    val_ds = CarDamageDataset(
        img_txt = os.path.join(args.split_dir, "val.txt"),
        val     = True,
        transform = tf
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    model = deeplabv3_resnet101(pretrained=True, num_classes=1)
    model.to(device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val    = float("inf")
    no_improve  = 0

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        v_loss  = validate(model, val_loader, device)
        print(f" Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f}")

        if v_loss < best_val:
            best_val   = v_loss
            no_improve = 0
            ckpt = os.path.join(args.output_dir, "best.pth")
            torch.save(model.state_dict(), ckpt)
            print(" Saved best", ckpt)
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    last_ckpt = os.path.join(args.output_dir, "last.pth")
    torch.save(model.state_dict(), last_ckpt)
    print("Training complete. Last model saved to", last_ckpt)


if __name__ == "__main__":
    main()
