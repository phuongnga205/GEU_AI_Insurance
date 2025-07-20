from flask import Flask, request, send_file
import torch
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

MODEL_PATH = "models/checkpoints/best_epoch2.pth"
device = 'cpu'

model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=1)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)   # <-- chỉ sửa dòng này!
model.to(device).eval()

def infer(img_pil):
    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    x = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)['out']
        pred = (out.squeeze() > 0).cpu().numpy()  # mask nhị phân
    mask_img = Image.fromarray((pred*255).astype(np.uint8))
    buf = io.BytesIO()
    mask_img.save(buf, format='PNG')
    buf.seek(0)
    return buf

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    img_file = request.files['image']
    img = Image.open(img_file.stream).convert('RGB')
    buf = infer(img)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
