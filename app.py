import os
import pickle
import tensorflow as tf
# <--- THAY ĐỔI 1: Chỉ import redirect và url_for, không cần session
from flask import Flask, request, render_template, send_file, redirect, url_for

# Import hàm chính từ file process.py
from process import run_inference 

# ==============================================================================
# ==== ✨ TẢI MODEL VÀ GALLERY MỘT LẦN KHI KHỞI ĐỘNG ✨ ====
# ==============================================================================
# (Giữ nguyên phần tải model và gallery)
GALLERY_FILE_PATH = "anchor_gallery.pkl"

print("🧠 Đang tải model MobileNetV2 vào bộ nhớ...")
try:
    inference_model = tf.keras.applications.MobileNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(224, 224, 3), 
        pooling='avg'
    )
    print("✅ Model đã sẵn sàng.")
except Exception as e:
    print(f"❌ Lỗi nghiêm trọng khi tải model TensorFlow: {e}")
    inference_model = None

print(f"📚 Đang tải bộ sưu tập anchor từ '{GALLERY_FILE_PATH}'...")
try:
    if not os.path.exists(GALLERY_FILE_PATH):
        raise FileNotFoundError(f"Không tìm thấy file gallery. Vui lòng chạy script 'create_anchor_gallery.py' trước.")
    
    with open(GALLERY_FILE_PATH, 'rb') as f:
        loaded_anchor_gallery = pickle.load(f)
    print("✅ Tải gallery thành công.")
except Exception as e:
    print(f"❌ Lỗi nghiêm trọng khi tải file gallery: {e}")
    loaded_anchor_gallery = None

# ==============================================================================
# ==== CẤU HÌNH FLASK ====
# ==============================================================================
app = Flask(__name__)
UPLOAD_FOLDER = "test"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# <--- THAY ĐỔI 2: Tạo một biến toàn cục để lưu trữ trạng thái
# Đây là một dictionary để lưu kết quả của lần xử lý gần nhất.
last_result_state = {}

@app.route('/')
def index():
    # Xóa kết quả cũ khi người dùng quay về trang chủ
    global last_result_state
    last_result_state = {}
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if inference_model is None or loaded_anchor_gallery is None:
        return "❌ Lỗi Server: Model hoặc Anchor Gallery chưa được tải thành công. Vui lòng kiểm tra log của server.", 500

    file = request.files.get('image')
    if not file:
        return "❌ Không có file nào được tải lên.", 400

    filename = file.filename 
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    print(f"🚀 Bắt đầu kiểm tra ảnh: {filename}")
    output_path, results = run_inference(upload_path, inference_model, loaded_anchor_gallery)

    if output_path is None:
        return "❌ Xảy ra lỗi trong quá trình xử lý ảnh.", 500

    # <--- THAY ĐỔI 3: Lưu kết quả vào biến toàn cục
    global last_result_state
    last_result_state['img_path'] = f"/result_image/{os.path.basename(output_path)}"
    last_result_state['results'] = results
    
    # Vẫn sử dụng Post/Redirect/Get để tránh lỗi khi reload
    return redirect(url_for('show_result'))


# <--- THAY ĐỔI 4: Route hiển thị kết quả sẽ đọc từ biến toàn cục
@app.route('/result')
def show_result():
    # Lấy dữ liệu từ biến toàn cục
    if not last_result_state:
        # Nếu không có dữ liệu (ví dụ: server vừa khởi động, người dùng chưa upload)
        # thì điều hướng về trang chủ.
        return redirect(url_for('index'))

    # Render template với dữ liệu đã lưu
    return render_template("result.html", 
                           img_path=last_result_state.get('img_path'), 
                           results=last_result_state.get('results'))


# Đổi tên route để rõ ràng hơn
@app.route('/result_image/<filename>')
def result_image(filename):
    safe_path = os.path.join("check", os.path.basename(filename))
    if os.path.exists(safe_path):
        return send_file(safe_path, mimetype='image/jpeg')
    return "Image not found", 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)