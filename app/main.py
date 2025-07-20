import os
import io
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

BASEDIR = os.path.dirname(__file__)  # .../GEU_AI_Insurance_2/app/web
PROJECT_ROOT = os.path.abspath(os.path.join(BASEDIR, '..', '..'))  # .../GEU_AI_Insurance_2
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'best_epoch2.pth')
print(f">> Loading checkpoint from {MODEL_PATH!r}, exists: {os.path.exists(MODEL_PATH)}")

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DEVICE = 'cpu'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def infer_mask(img_pil: Image.Image) -> Image.Image:
    tf = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    x = tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)['out']
        pred = (out.squeeze() > 0).cpu().numpy()
    return Image.fromarray((pred * 255).astype(np.uint8))

def calculate_damage_percent(mask_path: str) -> float:
    mask = np.array(Image.open(mask_path).convert('L'))
    pct = (mask > 128).sum() / mask.size * 100
    return round(pct, 2)

def damage_level(pct: float) -> str:
    if pct < 5:
        return "Mild"
    elif pct < 20:
        return "Moderate"
    else:
        return "Severe"

model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=1)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE).eval()

@app.route('/', methods=['GET'])
def index():
    history = sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True)
    return render_template('index.html',
                           original_image=None,
                           mask_image=None,
                           damage_percent=None,
                           damage_level=None,
                           history=history)

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files.get('image')
    if not f or f.filename == '' or not allowed_file(f.filename):
        return redirect(url_for('index'))

    fn = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    f.save(in_path)

    pil = Image.open(in_path).convert('RGB')
    mask_img = infer_mask(pil)

    base, ext = fn.rsplit('.', 1)
    mask_name = f"{base}_mask.{ext}"
    mask_path = os.path.join(app.config['RESULT_FOLDER'], mask_name)
    mask_img.save(mask_path)

    pct = calculate_damage_percent(mask_path)
    lvl = damage_level(pct)

    history = sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True)
    return render_template('index.html',
                           original_image=fn,
                           mask_image=mask_name,
                           damage_percent=pct,
                           damage_level=lvl,
                           history=history)

@app.route('/delete/<filename>')
def delete_file(filename):
    for folder in (UPLOAD_FOLDER, RESULT_FOLDER):
        p = os.path.join(folder, filename)
        if os.path.exists(p):
            os.remove(p)
    return redirect(url_for('index'))

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/static/results/<filename>')
def serve_result(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
