from flask import Flask, render_template, request, redirect, url_for, send_file
import requests
import os

app = Flask(__name__)

API_URL = "http://127.0.0.1:5000/predict"  # Địa chỉ API inference bạn đang chạy

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            # Lưu file tạm để gửi qua API
            img_path = os.path.join('uploads', img_file.filename)
            os.makedirs('uploads', exist_ok=True)
            img_file.save(img_path)

            with open(img_path, 'rb') as f:
                files = {'image': f}
                res = requests.post(API_URL, files=files)

            # Giả sử API trả về ảnh kết quả (mask hoặc overlay)
            if res.status_code == 200:
                result_path = os.path.join('uploads', 'result_' + img_file.filename)
                with open(result_path, 'wb') as out:
                    out.write(res.content)
                return render_template('result.html', img_file=img_file.filename, result_file='result_' + img_file.filename)
            else:
                return f"Lỗi khi gọi API: {res.text}"
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join('uploads', filename))

if __name__ == '__main__':
    app.run(port=8080, debug=True)
