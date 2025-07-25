from flask import Flask, request, render_template, send_file
from process import run_inference
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file:
        return "❌ Không có file nào được tải lên.", 400

    # Đọc ảnh trực tiếp từ file upload
    file_bytes = file.read()
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        return "❌ Ảnh không hợp lệ hoặc bị lỗi.", 400

    # Gọi xử lý trực tiếp bằng ảnh ndarray
    output_path, results = run_inference(img)

    # Trả kết quả cho giao diện
    return render_template("result.html", img_path="/result", results=results)

@app.route('/result')
def result_image():
    return send_file("check/test_result.jpg", mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
