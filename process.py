# process.py

import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ==============================================================================
# ==== ⚙️ CONFIG (Cấu hình cho việc kiểm tra) ====
# ==============================================================================
GALLERY_FILE_PATH = "anchor_gallery.pkl"
OUTPUT_FOLDER = "check"
RESIZED_SHAPE = (512, 1024)
# THAY ĐỔI: Thêm ngưỡng cho helmet
THRESHOLDS = {
    # "helmet": 0.80, # Ngưỡng cho mũ bảo hiểm
    "shirt": 0.75,
    "pants": 0.8,
    "left_glove": 0.80,
    "right_glove": 0.80,
    "left_shoe": 0.70,
    "right_shoe": 0.70,
    "left_arm": 0.6,
    "right_arm": 0.6,
    "nametag": 0.65,
    "default": 0.70
}
# THAY ĐỔI: Thêm "helmet" vào danh sách labels
labels = [ "nametag", "shirt", "pants", "left_glove", "right_glove", "left_shoe", "right_shoe", "left_arm", "right_arm"]
colors = {"pass": (0, 255, 0), "fail": (0, 0, 255), "missing": (128, 128, 128)}

# ==============================================================================
# ==== 🧠 HELPER FUNCTIONS (Các hàm xử lý cần thiết) ====
# ==============================================================================
def _preprocess_image_for_embedding(img):
    """Hàm phụ để tiền xử lý ảnh trước khi đưa vào model."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_rgb)

def extract_embedding(image_path, model):
    """Trích xuất embedding cho ảnh test, nhận model làm tham số."""
    if not image_path or not os.path.exists(image_path): return None
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, (224, 224))
    processed_img = _preprocess_image_for_embedding(img)
    return model.predict(np.expand_dims(processed_img, axis=0), verbose=0)

def crop_pose(image_path, save_folder):
    """Cắt các bộ phận từ ảnh dựa trên các điểm pose."""
    image = cv2.imread(image_path)
    if image is None: return {}, {}, None
    image = cv2.resize(image, RESIZED_SHAPE)
    h, w, _ = image.shape
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks: return {}, {}, image
    landmarks = results.pose_landmarks.landmark
    def get_point(lm): return int(lm.x * w), int(lm.y * h)
    crops, crop_paths = {}, {}
    def save_crop(label, x1, y1, x2, y2):
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            crop = image[y1:y2, x1:x2]
            path = os.path.join(save_folder, f"crop_{label}.jpg")
            cv2.imwrite(path, crop)
            crops[label] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            crop_paths[label] = path

    # --- MỚI: Logic crop mũ bảo hiểm (đồng bộ với file create_anchor_gallery.py) ---
    # head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # head_landmarks = [landmarks[i] for i in head_indices]
    # head_xs = [int(l.x * w) for l in head_landmarks]
    # head_ys = [int(l.y * h) for l in head_landmarks]
    
    # if head_xs and head_ys:
    #     fx1, fy1 = min(head_xs), min(head_ys)
    #     fx2, fy2 = max(head_xs), max(head_ys)
    #     fh = fy2 - fy1
    #     fw = fx2 - fx1
        
    #     helmet_x1 = fx1 - int(fw * 0.2)
    #     helmet_y1 = fy1 - int(fh * 1.0)
    #     helmet_x2 = fx2 + int(fw * 0.2)
    #     helmet_y2 = fy2
    #     save_crop("helmet", helmet_x1, helmet_y1, helmet_x2, helmet_y2)

    # --- Logic crop các bộ phận khác (giữ nguyên) ---
    ls, rs = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lw, rw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    la, ra = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    lh, rh = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    x1, y1 = get_point(ls); x2, y2 = get_point(rs)
    cx1 = int((x1 + x2) * 0.5); cx2 = max(x1, x2) + 20
    cy1 = int((y1 + y2) * 0.5 + 0.08 * h)-20; cy2 = cy1 + 100
    save_crop("nametag", cx1, cy1, cx2, cy2)
    def crop_hand(label, ids):
        pts = [landmarks[i] for i in ids]
        xs = [int(p.x * w) for p in pts]; ys = [int(p.y * h) for p in pts]
        margin_x, margin_y = 30, 50
        save_crop(label, min(xs) - margin_x, min(ys) - margin_y, max(xs) + margin_x, max(ys) + margin_y)
    crop_hand("left_glove", [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.LEFT_INDEX])
    crop_hand("right_glove", [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY, mp_pose.PoseLandmark.RIGHT_INDEX])
    for label, pt in zip(["left_shoe", "right_shoe"], [la, ra]):
        px, py = get_point(pt)
        save_crop(label, px - 50, py - 20, px + 50, py + 60)
    x_ls, y_ls = get_point(ls); x_rs, y_rs = get_point(rs)
    shirt_x1 = min(x_ls, x_rs) - 20; shirt_y1 = min(y_ls, y_rs) - 40
    shirt_x2 = max(x_ls, x_rs) + 20; shirt_y2 = int((lh.y + rh.y) / 2 * h)
    save_crop("shirt", shirt_x1, shirt_y1, shirt_x2, shirt_y2)
    lx, ly = get_point(lh); rx, ry = get_point(rh)
    ankle_y = max(get_point(la)[1], get_point(ra)[1])
    save_crop("pants", min(lx, rx) - 80, min(ly, ry), max(lx, rx) + 80, ankle_y + 40)
    for label, shoulder, wrist in zip(["left_arm", "right_arm"], [ls, rs], [lw, rw]):
        sx, sy = get_point(shoulder); wx, wy = get_point(wrist)
        save_crop(label, min(sx, wx) - 30, min(sy, wy) - 30, max(sx, wx) + 30, max(sy, wy) + 30)
    return crops, crop_paths, image

# ==============================================================================
# ==== 🚀 MAIN INFERENCE FUNCTION (TỐI ƯU) 🚀 ====
# ==============================================================================
def run_inference(test_image_path, model, anchor_gallery):
    """Hàm kiểm tra chính, nhận model và gallery đã tải làm tham số."""
    
    # --- BƯỚC 1: XỬ LÝ ẢNH TEST ---
    print("🔧 Đang crop ảnh test...")
    os.makedirs(f"{OUTPUT_FOLDER}/test", exist_ok=True)
    test_boxes, test_paths, test_image = crop_pose(test_image_path, f"{OUTPUT_FOLDER}/test")
    if test_image is None:
        return None, {"error": "Failed to process test image."}

    # --- BƯỚC 2: SO SÁNH VÀ ĐÁNH GIÁ ---
    # Vòng lặp này giờ sẽ tự động xử lý "helmet" vì nó đã có trong `labels`
    results = {}
    for label in labels:
        result = "missing"
        emb_test = extract_embedding(test_paths.get(label), model)
        gallery = anchor_gallery.get(label)

        if emb_test is not None and gallery:
            similarities = [cosine_similarity(emb_test, emb_anchor)[0][0] for emb_anchor in gallery]
            max_sim = max(similarities)
            current_threshold = THRESHOLDS.get(label, THRESHOLDS["default"])
            
            print(f"[{label.upper():<12}] - Max Similarity: {max_sim:.4f} (Threshold: {current_threshold})")
            result = "pass" if max_sim >= current_threshold else "fail"
        else:
            print(f"[{label.upper():<12}] - Missing test crop or anchor gallery for this label.")
        
        # --- LOGIC KIỂM TRA PHỤ ---
        if result == "pass" and label in ["pants", "left_arm", "right_arm"]:
            path = test_paths.get(label)
            if path and os.path.exists(path):
                img = cv2.imread(path)
                h_img = img.shape[0]
                start_row = int(h_img * (2/3 if label == 'pants' else 1/2))
                check_area = img[start_row:, :]
                if check_area.size > 0:
                    hsv = cv2.cvtColor(check_area, cv2.COLOR_BGR2HSV)
                    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                    mask = cv2.inRange(hsv, lower_skin, upper_skin)
                    skin_ratio = np.sum(mask == 255) / mask.size if mask.size > 0 else 0
                    print(f"  -> [{label.upper()}] Skin check. Ratio: {skin_ratio:.2%}")
                    if skin_ratio > 0.05:
                        print(f"  -> !!! Phát hiện da, kết quả cho '{label}' chuyển thành FAIL.")
                        result = "fail"
        results[label] = result

        # --- VẼ KẾT QUẢ LÊN ẢNH ---
        if result == "fail" and label in test_boxes:
            box = test_boxes[label]
            cv2.rectangle(test_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), colors["fail"], 2)
            cv2.putText(test_image, f"{label}: fail", (box["x1"], box["y1"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["fail"], 2)

    # --- LƯU KẾT QUẢ ---
    output_path = os.path.join(OUTPUT_FOLDER, "test_result.jpg")
    cv2.imwrite(output_path, test_image)
    
    return output_path, results

if __name__ == '__main__':
    # BẠN HÃY THAY ĐƯỜNG DẪN NÀY
    TEST_IMAGE_FILE = "path/to/your/test_image.jpg" 
    
    if not os.path.exists(GALLERY_FILE_PATH):
        print(f"❌ Lỗi: Không tìm thấy file gallery '{GALLERY_FILE_PATH}'.")
        print("➡️ Vui lòng chạy script `create_anchor_gallery.py` trước để tạo file này.")
    elif not os.path.exists(TEST_IMAGE_FILE):
        print(f"❌ Lỗi: Không tìm thấy file ảnh test tại '{TEST_IMAGE_FILE}'")
        print("➡️ Vui lòng cập nhật biến TEST_IMAGE_FILE trong code.")
    else:
        print("🧠 Đang tải model MobileNetV2...")
        inference_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
        print("✅ Model đã sẵn sàng.")
        
        print(f"📚 Đang tải bộ sưu tập anchor từ '{GALLERY_FILE_PATH}'...")
        with open(GALLERY_FILE_PATH, 'rb') as f:
            loaded_anchor_gallery = pickle.load(f)
        print("✅ Tải gallery thành công.")

        print("\n🚀 Bắt đầu kiểm tra ảnh test...")
        output, final_results = run_inference(TEST_IMAGE_FILE, inference_model, loaded_anchor_gallery)
        
        if output:
            print(f"\n✅ Xử lý hoàn tất. Kết quả được lưu tại: {output}")
            print("================== KẾT QUẢ KIỂM TRA =================")
            print(json.dumps(final_results, indent=2, ensure_ascii=False))
            print("=====================================================")