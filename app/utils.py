import os
import io
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# --- CẤU HÌNH MODEL ---
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(BASE, 'models', 'checkpoints', 'best_epoch2.pth')
DEVICE = 'cpu'

# --- LOAD MODEL 1 LẦN ---
model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=1)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()

# --- TRANSFORMS ---
_tf = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

def infer_mask(img: Image.Image) -> Image.Image:
    """Từ PIL Image trả về PIL mask B/W."""
    x = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)['out']
        pred = (out.squeeze() > 0).cpu().numpy()
    return Image.fromarray((pred*255).astype(np.uint8))

def calculate_damage_percent(mask: Image.Image) -> float:
    arr = np.array(mask.convert('L'))
    pct = (arr>128).sum() / arr.size * 100
    return round(pct, 2)

def damage_level(pct: float) -> str:
    if pct < 5:   return "Mild"
    if pct <20:   return "Moderate"
    return "Severe"

def process_image(input_path: str, output_folder: str):
    """
    1) Đọc ảnh gốc → PIL
    2) infer mask
    3) save mask vào output_folder
    4) tính % + level
    Trả về: (mask_filename, percent, level)
    """
    img = Image.open(input_path).convert('RGB')
    mask = infer_mask(img)

    name, ext = os.path.splitext(os.path.basename(input_path))
    mask_name = f"{name}_mask{ext}"
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, mask_name)
    mask.save(path)

    pct = calculate_damage_percent(mask)
    lvl = damage_level(pct)
    return mask_name, pct, lvl
