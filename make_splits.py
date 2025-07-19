import os, random

# 1. Lấy danh sách tên file (không có đuôi)
img_dir = "sample_car_damage"
names = [os.path.splitext(f)[0] for f in os.listdir(img_dir)]
random.seed(42)
random.shuffle(names)

# 2. Phân chia 70% train, 15% val, 15% test
n = len(names)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

splits = {
    "train.txt": names[:n_train],
    "val.txt":   names[n_train:n_train + n_val],
    "test.txt":  names[n_train + n_val:],
}

# 3. Ghi ra file .txt
for fn, lst in splits.items():
    with open(fn, "w") as f:
        f.write("\n".join(lst))
    print(f"Wrote {len(lst)} entries to {fn}")
