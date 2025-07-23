import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
from glob import glob

# ==== ⚙️ CONFIG ====
INPUT_FOLDER = "anchors"  # Thư mục chứa các ảnh anchor
OUTPUT_PKL = "embeddings.pkl"    # File để lưu embedding
RESIZED_SHAPE = (512, 1024)      # Kích thước ảnh crop, đồng bộ với process.py

# ==== 🧠 MODEL ====
model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                          input_shape=(224, 224, 3), pooling='avg')

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)[0]  # Trả về embedding dạng 1D

# Hàm crop_pose (sao chép từ process.py để đảm bảo đồng bộ)
def crop_pose(image_path, save_folder):
    image = cv2.imread(image_path)
    image = cv2.resize(image, RESIZED_SHAPE)
    h, w, _ = image.shape
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        print(f"❌ Không phát hiện người trong ảnh: {image_path}")
        return {}, {}, image

    landmarks = results.pose_landmarks.landmark

    def get_point(lm):
        return int(lm.x * w), int(lm.y * h)

    crops = {}
    crop_paths = {}

    def save_crop(label, x1, y1, x2, y2):
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            crop = image[y1:y2, x1:x2]
            path = os.path.join(save_folder, f"crop_{label}.jpg")
            cv2.imwrite(path, crop)
            crops[label] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            crop_paths[label] = path

    # Điểm landmark chính
    ls, rs = landmarks[11], landmarks[12]
    lw, rw = landmarks[15], landmarks[16]
    la, ra = landmarks[27], landmarks[28]
    lh, rh = landmarks[23], landmarks[24]

    # Nametag
    x1, y1 = get_point(ls)
    x2, y2 = get_point(rs)
    cx1 = int((x1 + x2) * 0.5)
    cx2 = max(x1, x2) + 20
    cy1 = int((y1 + y2) * 0.5 + 0.08 * h) - 20
    cy2 = cy1 + 100
    save_crop("nametag", cx1, cy1, cx2, cy2)

    # Găng tay
    def crop_hand(label, ids):
        pts = [landmarks[i] for i in ids]
        xs = [int(p.x * w) for p in pts]
        ys = [int(p.y * h) for p in pts]
        margin_x, margin_y = 30, 50
        save_crop(label, min(xs) - margin_x, min(ys) - margin_y, max(xs) + margin_x, max(ys) + margin_y)

    crop_hand("left_glove", [15, 17, 19, 21])
    crop_hand("right_glove", [16, 18, 20, 22])

    # Giày
    for label, pt in zip(["left_shoe", "right_shoe"], [la, ra]):
        px, py = get_point(pt)
        save_crop(label, px - 50, py - 20, px + 50, py + 60)

    # Áo
    x_ls, y_ls = get_point(ls)
    x_rs, y_rs = get_point(rs)
    shirt_x1 = min(x_ls, x_rs) - 20
    shirt_y1 = min(y_ls, y_rs) - 40
    shirt_x2 = max(x_ls, x_rs) + 20
    shirt_y2 = int((lh.y + rh.y) / 2 * h)
    save_crop("shirt", shirt_x1, shirt_y1, shirt_x2, shirt_y2)

    # Quần
    lx, ly = get_point(lh)
    rx, ry = get_point(rh)
    ankle_y = max(get_point(la)[1], get_point(ra)[1])
    save_crop("pants", min(lx, rx) - 80, min(ly, ry), max(lx, rx) + 80, ankle_y + 40)

    # Cánh tay
    for label, shoulder, wrist in zip(["left_arm", "right_arm"], [ls, rs], [lw, rw]):
        sx, sy = get_point(shoulder)
        wx, wy = get_point(wrist)
        save_crop(label, min(sx, wx) - 30, min(sy, wy) - 30, max(sx, wx) + 30, max(sy, wy) + 30)

    return crops, crop_paths, image

# Duyệt qua tất cả ảnh trong thư mục
embeddings_dict = {}
image_files = glob(os.path.join(INPUT_FOLDER, "*.jpg"))  # Giả sử ảnh là .jpg

for image_path in image_files:
    file_name = os.path.basename(image_path)
    temp_folder = f"temp_crops/{file_name}"  # Thư mục tạm để lưu các crop
    os.makedirs(temp_folder, exist_ok=True)

    # Crop ảnh anchor
    _, crop_paths, _ = crop_pose(image_path, temp_folder)

    # Trích xuất embedding cho từng vùng crop
    embeddings_dict[file_name] = {}
    for label in ["shirt", "pants", "left_glove", "right_glove", "left_shoe", "right_shoe", "left_arm", "right_arm", "nametag"]:
        crop_path = crop_paths.get(label)
        if crop_path:
            embedding = extract_embedding(crop_path)
            if embedding is not None:
                embeddings_dict[file_name][label] = embedding
                print(f"Đã trích xuất embedding cho {file_name} - {label}")
            else:
                print(f"Không thể trích xuất embedding cho {file_name} - {label}")

# Lưu embedding vào file .pkl
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f"Đã lưu embedding vào {OUTPUT_PKL}")