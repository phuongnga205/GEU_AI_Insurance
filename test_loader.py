import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import CarDamageDataset

if __name__ == "__main__":
    # 1) đọc list train.txt
    with open("train.txt") as f:
        names = [l.strip() for l in f]

    # 2) transform cho image: resize, to tensor, normalize
    img_transform = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225]),
    ])

    # 3) transform cho mask: resize (nearest), to tensor
    mask_transform = T.Compose([
        T.Resize((520, 520), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),   # giá trị sẽ nằm [0,1]
    ])

    # 4) khởi tạo dataset & dataloader
    ds = CarDamageDataset(
        img_dir="sample_car_damage",
        mask_dir="annotations",
        names_list=names,
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    # 5) lấy 1 batch và in shape
    images, masks = next(iter(loader))
    print("Ảnh batch:", images.shape)   # ex: [4,3,520,520]
    print("Mask batch:", masks.shape)    # ex: [4,1,520,520]
