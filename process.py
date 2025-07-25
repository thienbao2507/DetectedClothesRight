import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load mô hình helmet
helmet_model = YOLO("best2.pt")
helmet_model.eval()

# ==== ⚙️ CONFIG ====
OUTPUT_FOLDER = "check"
RESIZED_SHAPE = (512, 1024)

# ==== THRESHOLDS & COLOR RANGES ====

# HSV Ranges
HSV_SKIN = (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
HSV_SHOE = (np.array([0, 40, 90]), np.array([18, 150, 255]))

# Thresholds
THRESH_SKIN_SHOE = 0.06
THRESH_SKIN_GLOVE_FULL = 0.4
THRESH_SKIN_GLOVE_TIP = 0.02
THRESH_SKIN_ARM = 0.1
THRESH_SKIN_PANTS = 0.02
THRESH_HELMET_CONF = 0.70
THRESH_SMILE_CONF = 0.5
THRESH_NAMETAG_BRIGHT = 170
THRESH_NAMETAG_RATIO = 0.03
THRESH_NAMETAG_AREA = 300



# Load mô hình phát hiện nụ cười
smile_model = tf.keras.models.load_model("Smile_Detection/Smile_Detection/output/smile.h5")  # đổi tên nếu khác
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


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
    return model.predict(img, verbose=0)


# ==== 📦 LABELS + COLOR ====
labels = ["nametag", "shirt", "pants", "left_glove", "right_glove",
          "left_shoe", "right_shoe", "left_arm", "right_arm"]
colors = {"pass": (0, 255, 0), "fail": (0, 0, 255), "missing": (128, 128, 128)}


# ===NAMETAG===
def detect_nametag_better(image_input, bright_threshold=THRESH_NAMETAG_BRIGHT,
                           ratio_thresh=THRESH_NAMETAG_RATIO,
                           area_thresh=THRESH_NAMETAG_AREA, show=True):
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            print("❌ File không tồn tại:", image_input)
            return "missing", None
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        print("❌ Input không hợp lệ (không phải path hoặc ảnh)")
        return "missing", None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold để tìm vùng sáng (thẻ tên)
    _, binary = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)

    # Tìm contours để xác định vùng sáng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_ratio = 0.0  # Khởi tạo white_ratio
    largest_area = 0
    best_box = None
    found = False

    # Tìm contour lớn nhất
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(cnt)
            best_box = (x, y, x + w, y + h)
            found = area > area_thresh

    # Tính tỷ lệ pixel sáng dựa trên contour lớn nhất
    if largest_area > 0.5:
        white_ratio = largest_area / binary.size
        print(f"🔍 Bright pixel ratio (largest cluster): {white_ratio:.2%}")

    if show and found:
        cv2.rectangle(img, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
        cv2.putText(img, f"Area: {int(largest_area)}", (best_box[0], best_box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return ("pass" if (white_ratio > ratio_thresh or found) else "fail"), best_box


import cv2
import numpy as np


def evaluate_shirt_color_hsv_direct(img, save_path=None):
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Ngưỡng màu
    lower_orange = np.array([0, 70, 70])
    upper_orange = np.array([30, 255, 255])
    blue_range = (np.array([95, 30, 35]), np.array([135, 255, 255]))

    # ROI phần ngực (sọc cam)
    top = int(h_img * 0.18)
    bottom = int(h_img * 0.42)
    left = int(w_img * 0.05)
    right = int(w_img * 0.95)
    roi = hsv[top:bottom, left:right]

    kernel = np.ones((5, 5), np.uint8)
    roi_mask = cv2.inRange(roi, lower_orange, upper_orange)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = roi.shape[0] * roi.shape[1]

    if not contours:
        if save_path:
            debug_img = img.copy()
            cv2.putText(debug_img, "No orange detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(save_path, debug_img)
        return "fail"

    largest_cnt = max(contours, key=cv2.contourArea)
    cam_area = cv2.contourArea(largest_cnt)

    # Kiểm tra chặt hơn: diện tích + tỉ lệ + mật độ pixel cam
    x, y, w_box, h_box = cv2.boundingRect(largest_cnt)
    aspect_ratio = w_box / h_box if h_box > 0 else 0
    cam_pixel_ratio = np.sum(roi_mask > 0) / roi_area

    if (cam_area / roi_area < 0.02) or (aspect_ratio < 2.0) or (cam_pixel_ratio < 0.01):
        if save_path:
            debug_img = img.copy()
            cv2.putText(debug_img, "CAM too small/short/insufficient", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(save_path, debug_img)
        return "fail"

    # Xác định lại vùng cam tuyệt đối
    x_abs = x + left
    y_abs = y + top
    cam_roi = hsv[y_abs:y_abs + h_box, x_abs:x_abs + w_box]
    cam_mask = cv2.inRange(cam_roi, lower_orange, upper_orange)
    cam_mean = np.array(cv2.mean(cam_roi, mask=cam_mask)[:3])

    # Kiểm tra vùng xanh bên dưới
    bot_hsv = hsv[y_abs + h_box:, x_abs:x_abs + w_box]
    bot_mean = np.array(cv2.mean(bot_hsv)[:3])

    def in_range(color, color_range):
        lower, upper = color_range
        return np.all(color >= lower) and np.all(color <= upper)

    cam_match = np.sum(cam_mask > 0) > 0
    bottom_match = in_range(bot_mean, blue_range)

    result = "pass" if cam_match and bottom_match else "fail"

    # Vẽ debug nếu cần
    if save_path:
        debug_img = img.copy()

        # CAM
        cv2.rectangle(debug_img, (x_abs, y_abs), (x_abs + w_box, y_abs + h_box), (0, 165, 255), 2)
        cv2.putText(debug_img, "CAM", (x_abs, y_abs - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 165, 255), 1)

        # BLUE
        cv2.rectangle(debug_img, (x_abs, y_abs + h_box), (x_abs + w_box, h_img), (255, 0, 0), 2)
        cv2.putText(debug_img, "BLUE", (x_abs, y_abs + h_box + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)

        if result == "fail":
            if not cam_match:
                cv2.putText(debug_img, "❌ Sai CAM", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not bottom_match:
                cv2.putText(debug_img, "❌ Sai BLUE (Duoi)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(debug_img, "✅ Dung mau dong phuc", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imwrite(save_path, debug_img)

    return result


# ==== 📌 POSE CROP ====
def crop_pose(image_path, save_folder):
    if isinstance(image_path, np.ndarray):
        image = image_path
    else:
        image = cv2.imread(image_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print(f"❌ Không phát hiện người trong ảnh: {image_path}")
        return {}, {}, image

    landmarks = results.pose_landmarks.landmark
    h_raw, w_raw = image.shape[:2]
    points = [(int(lm.x * w_raw), int(lm.y * h_raw)) for lm in landmarks]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    margin_x = 350
    margin_y = 500

    x1 = max(min(xs) - margin_x, 0)
    y1 = max(min(ys) - margin_y, 0)
    x2 = min(max(xs) + margin_x, w_raw)
    y2 = min(max(ys) + margin_y, h_raw)

    # Cắt vùng chứa người
    image = image[y1:y2, x1:x2]
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
    crop_images = {}

    def save_crop(label, x1, y1, x2, y2):
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            crop = image[y1:y2, x1:x2]
            crops[label] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            crop_images[label] = crop

    # Điểm landmark chính
    ls, rs = landmarks[11], landmarks[12]
    lw, rw = landmarks[15], landmarks[16]
    la, ra = landmarks[27], landmarks[28]
    lh, rh = landmarks[23], landmarks[24]

    # Nametag
    x1, y1 = get_point(ls)
    x2, y2 = get_point(rs)
    cx1 = int((x1 + x2) * 0.5)
    cx2 = max(x1, x2) + 10
    cy1 = int((y1 + y2) * 0.5 + 0.08 * h) - 20
    cy2 = cy1 + 100  # độ dài
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
    save_crop("pants", min(lx, rx) - 80, min(ly, ry), max(lx, rx) + 80, ankle_y - 10)

    # Cánh tay
    for label, shoulder, wrist in zip(["left_arm", "right_arm"], [ls, rs], [lw, rw]):
        sx, sy = get_point(shoulder)
        wx, wy = get_point(wrist)
        save_crop(label, min(sx, wx) - 30, min(sy, wy) - 30, max(sx, wx) + 30, max(sy, wy) + 30)

    # === CROP VÙNG ĐẦU (HELMET) ===
    head_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # mũi, mắt, miệng, tai
    head_points = [get_point(landmarks[i]) for i in head_ids]
    xs = [p[0] for p in head_points]
    ys = [p[1] for p in head_points]

    # Mở rộng vùng đầu để lấy cả nón
    margin_x, margin_y = 60, 70
    extra_top_margin = 30  # ✅ mở rộng thêm lên phía trên

    x1 = max(min(xs) - margin_x, 0)
    y1 = max(min(ys) - margin_y - extra_top_margin, 0)
    x2 = min(max(xs) + margin_x, w)
    y2 = min(max(ys) + margin_y, h)

    save_crop("helmet", x1, y1, x2, y2)

    # === CROP VÙNG MẶT NHỎ HƠN CHỈ DÙNG NHẬN NỤ CƯỜI ===
    # === CROP VÙNG MẶT (RIÊNG CHO NHẬN DIỆN NỤ CƯỜI) ===
    face_margin_x = 20
    top_margin = 40
    bottom_margin = 70  # 👉 mở rộng thêm phía dưới để chắc chắn có miệng

    fx1 = max(min(xs) - face_margin_x, 0)
    fy1 = max(min(ys) - top_margin, 0)
    fx2 = min(max(xs) + face_margin_x, w)
    fy2 = min(max(ys) + bottom_margin, h)  # 👉 mở rộng xuống dưới

    save_crop("face_smile", fx1, fy1, fx2, fy2)

    return crops, crop_images, image, landmarks


def extract_shirt_colors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (300, 300))  # đảm bảo tỷ lệ chuẩn

    h = img.shape[0]
    top = img[0:int(h / 3), :, :]
    mid = img[int(h / 3):int(2 * h / 3), :, :]
    bot = img[int(2 * h / 3):, :, :]

    color_top = np.mean(top.reshape(-1, 3), axis=0)
    color_mid = np.mean(mid.reshape(-1, 3), axis=0)
    color_bot = np.mean(bot.reshape(-1, 3), axis=0)

    return {
        "top": color_top,
        "mid": color_mid,
        "bot": color_bot
    }


def detect_smile(image_input, threshold=THRESH_SMILE_CONF):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return "missing"

    if img is None:
        return "missing"


    detector = cv2.CascadeClassifier("Smile_Detection/Smile_Detection/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(rects) == 0:
        return "no_face"

    (fX, fY, fW, fH) = rects[0]
    roi_color = img[fY:fY + fH, fX:fX + fW]
    roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (224, 224))

    array_img = img_to_array(roi_resized)
    array_img = preprocess_input(array_img)

    # landmark môi
    results = face_mesh.process(roi_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left = landmarks[61]
        right = landmarks[291]
        top = landmarks[13]
        bottom = landmarks[14]
        mouth_width = np.linalg.norm(np.array([left.x, left.y]) - np.array([right.x, right.y]))
        mouth_height = np.linalg.norm(np.array([top.y]) - np.array([bottom.y]))
    else:
        mouth_width, mouth_height = 0.0, 0.0

    combined = np.append(array_img.flatten(), [mouth_width, mouth_height])
    combined = np.expand_dims(combined, axis=0)

    (not_smile, smile) = smile_model.predict(combined, verbose=0)[0]
    print(f"🙂 Smile confidence: {smile:.2%} | Not smile: {not_smile:.2%}")

    return "smile" if smile > threshold else "no_smile"


def intersect_with_leg_line(box, knee, ankle):
    """
    Kiểm tra xem bounding box có cắt qua đường thẳng từ đầu gối đến gót chân không.

    Args:
        box: tuple (x1, y1, x2, y2) – toạ độ vùng da
        knee: tuple (x, y) – toạ độ đầu gối
        ankle: tuple (x, y) – toạ độ gót chân

    Returns:
        True nếu cắt qua, False nếu nằm lệch ngoài
    """
    x1, y1, x2, y2 = box
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Tọa độ điểm đầu và cuối đường trục chân
    x_knee, y_knee = knee
    x_ankle, y_ankle = ankle

    # Duyệt theo chiều y, kiểm tra từng điểm trên đường trục
    for alpha in np.linspace(0, 1, 20):  # kiểm tra 20 điểm trên đoạn thẳng
        x_line = int((1 - alpha) * x_knee + alpha * x_ankle)
        y_line = int((1 - alpha) * y_knee + alpha * y_ankle)

        if x_min <= x_line <= x_max and y_min <= y_line <= y_max:
            return True  # Có giao

    return False  # Không cắt qua


# ==DeLoy==
def run_inference(test_image_path):
    # Tạo lại thư mục kết quả
    os.makedirs(f"{OUTPUT_FOLDER}/anchor", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/test", exist_ok=True)
    box_errors = []

    # ==== 📌 POSE CROP ====
    print("🔧 Đang crop ảnh test...")
    test_boxes, test_crops, test_image, test_landmarks = crop_pose(test_image_path, f"{OUTPUT_FOLDER}/test")

    results = {}
    early_fail = False
    all_labels = labels.copy()

    for label in all_labels:

        if label in ["left_arm", "right_arm"]:
            continue
        if label in ["left_glove", "right_glove"]:
            img = test_crops.get(label)
            if img is None:
                result = "missing"
            else:
                # === Kiểm tra toàn bàn tay
                hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask_full = cv2.inRange(hsv_full, np.array([0, 20, 70], dtype=np.uint8),
                                        np.array([20, 255, 255], dtype=np.uint8))
                skin_ratio_full = np.sum(mask_full == 255) / mask_full.size
                print(f"[{label.upper()}] skin ratio (full): {skin_ratio_full:.2%}")

                # === Kiểm tra đầu ngón tay (1/3 dưới)
                h = img.shape[0]
                roi = img[int(h * 2 / 3):, :]
                hsv_tip = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask_tip = cv2.inRange(hsv_tip, np.array([0, 20, 70], dtype=np.uint8),
                                       np.array([20, 255, 255], dtype=np.uint8))
                skin_ratio_tip = np.sum(mask_tip == 255) / mask_tip.size
                print(f"[{label.upper()}] skin ratio (fingertips): {skin_ratio_tip:.2%}")

                if skin_ratio_full > THRESH_SKIN_GLOVE_FULL:
                    result = "fail"
                    if label in test_boxes:
                        box = test_boxes[label]
                        box_errors.append({
                            "label": f"{label}_no_glove",
                            "box": (box["x1"], box["y1"], box["x2"], box["y2"]),
                            "color": (0, 0, 255)
                        })
                elif skin_ratio_tip > THRESH_SKIN_GLOVE_TIP:
                    result = "fail"
                    if label in test_boxes:
                        contours, _ = cv2.findContours(mask_tip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 100:
                                continue
                            x, y, w, h = cv2.boundingRect(cnt)
                            box = test_boxes[label]
                            x1 = box["x1"] + x
                            y1 = box["y1"] + int((box["y2"] - box["y1"]) * 2 / 3) + y
                            x2 = x1 + w
                            y2 = y1 + h
                            box_errors.append({
                                "label": f"{label}_tip_skin",
                                "box": (x1, y1, x2, y2),
                                "color": (0, 0, 255)
                            })
                else:
                    result = "pass"

            results[label] = result

        if label == "nametag":
            if early_fail:
                results[label] = "fail"
                continue
            img = test_crops.get(label)
            if img is not None:
                result, nametag_box = detect_nametag_better(img)
            else:
                result = "missing"
                nametag_box = None
            offset = test_boxes["nametag"]
            if nametag_box:
                x1_crop, y1_crop, x2_crop, y2_crop = nametag_box
                x1 = offset["x1"] + x1_crop
                y1 = offset["y1"] + y1_crop
                x2 = offset["x1"] + x2_crop
                y2 = offset["y1"] + y2_crop
            results[label] = result

        else:
            if label == "shirt":
                shirt_img = test_crops.get("shirt")
                result = "missing"
                if shirt_img is not None:
                    result = evaluate_shirt_color_hsv_direct(shirt_img)
                    results["shirt"] = result
            if label in ["shirt", "pants"] and result == "fail":
                early_fail = True
                # Nếu pants là pass, kiểm tra xem có bị sắn (lộ da) không
            if label == "pants":
                img = test_crops.get("pants")
                if img is not None:
                    result = "pass"
                    h = img.shape[0]
                    start_row = int(h * 1 / 2)
                    lower_part = img[start_row:, :]

                    hsv = cv2.cvtColor(lower_part, cv2.COLOR_BGR2HSV)
                    lower = np.array([0, 20, 70], dtype=np.uint8)
                    upper = np.array([20, 255, 255], dtype=np.uint8)
                    mask = cv2.inRange(hsv, lower, upper)
                    skin_ratio = np.sum(mask == 255) / mask.size
                    print(f"[PANTS SẮN] Skin ratio (lower): {skin_ratio:.2%}")

                    if skin_ratio > THRESH_SKIN_PANTS:
                        # ======= BỔ SUNG: In ra vị trí vùng da so với đầu gối và gót chân ========
                        def get_point(lm):  # Convert landmark to pixel
                            return int(lm.x * test_image.shape[1]), int(lm.y * test_image.shape[0])

                        left_knee = get_point(test_landmarks[25])
                        right_knee = get_point(test_landmarks[26])
                        left_ankle = get_point(test_landmarks[27])
                        right_ankle = get_point(test_landmarks[28])

                        print(f"LEFT_KNEE: {left_knee}")
                        print(f"RIGHT_KNEE: {right_knee}")
                        print(f"LEFT_ANKLE: {left_ankle}")
                        print(f"RIGHT_ANKLE: {right_ankle}")

                        if label in test_boxes:
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                if cv2.contourArea(cnt) < 50:
                                    continue
                                x, y, w, h_cnt = cv2.boundingRect(cnt)
                                box = test_boxes["pants"]
                                x1 = box["x1"] + x
                                y1 = box["y1"] + start_row + y
                                x2 = x1 + w
                                y2 = y1 + h_cnt
                                region_box = (x1, y1, x2, y2)

                                # 🧠 Kiểm tra có giao với trục chân không
                                if intersect_with_leg_line(region_box, left_knee, left_ankle) or \
                                        intersect_with_leg_line(region_box, right_knee, right_ankle):
                                    print("✅ Vùng da giao với chân → là lỗi thật")
                                    test_boxes["pants_rolled_up"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                    all_labels.append("pants_rolled_up")
                                    results["pants_rolled_up"] = "fail"
                                    box_errors.append({
                                        "label": "pants_rolled_up",
                                        "box": (x1, y1, x2, y2),
                                        "color": (0, 0, 255)
                                    })
                                else:
                                    print("❌ Bỏ qua vùng da không nằm trên chân")

                results["pants"] = result
        if label in ["left_shoe", "right_shoe"]:
            img = test_crops.get(label)
            if img is None:
                result = "missing"
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 40, 90])
                upper = np.array([18, 150, 255])
                mask = cv2.inRange(hsv, lower, upper)
                skin_ratio = np.sum(mask == 255) / mask.size
                print(f"[{label.upper()}] Skin ratio (shoe): {skin_ratio:.2%}")
                result = "fail" if skin_ratio > THRESH_SKIN_SHOE  else "pass"
            results[label] = result
        if label == "shirt" and result == "pass":
            for arm_label in ["left_arm", "right_arm"]:
                img = test_crops.get(arm_label)
                if img is None:
                    results[arm_label] = "missing"
                    continue

                # Cắt 2/3 dưới ảnh tay để tránh vùng vai
                h = img.shape[0]
                roi = img[int(h / 3):, :]  # từ 1/3 chiều cao trở xuống

                # Xử lý HSV trên ROI
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 20, 70], dtype=np.uint8)
                upper = np.array([20, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                skin_ratio = np.sum(mask == 255) / mask.size

                print(f"[{arm_label.upper()}] skin ratio: {skin_ratio:.2%}")

                if skin_ratio > THRESH_SKIN_ARM :
                    results[arm_label] = "fail"
                    if arm_label in test_boxes:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 100:
                                continue
                            x, y, w, h_box = cv2.boundingRect(cnt)
                            box = test_boxes[arm_label]
                            x1 = box["x1"] + x
                            y1 = box["y1"] + int(h / 3) + y
                            x2 = x1 + w
                            y2 = y1 + h_box
                            box_errors.append({
                                "label": f"{arm_label}_skin",
                                "box": (x1, y1, x2, y2),
                                "color": (0, 0, 255)
                            })
                else:
                    results[arm_label] = "pass"

        # results[label] = result
        DRAW_FAIL_LABELS = [
            "nametag", "left_glove", "right_glove",
            "pants", "pants_rolled_up", "left_arm_skin", "right_arm_skin",
            "left_shoe", "right_shoe", "shirt", "helmet"
        ]
        # 🎨 Vẽ khung lên ảnh
        if result == "fail" and label in DRAW_FAIL_LABELS and label in test_boxes:
            color = colors["fail"]
            box = test_boxes[label]
            cv2.rectangle(test_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
            cv2.putText(test_image, f"{label}: {result}", (box["x1"], box["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # === Vẽ thêm tất cả các lỗi phụ từ box_errors ===
        for error in box_errors:
            err_label = error["label"]
            if err_label not in DRAW_FAIL_LABELS:
                continue  # bỏ qua các lỗi không cần vẽ
            x1, y1, x2, y2 = error["box"]
            color = error.get("color", (0, 0, 255))
            cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(test_image, err_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # ==== HELMET CHECK ====
    helmet_path = test_image_path
    if helmet_path is not None:
        results_helmet = helmet_model(helmet_path)[0]  # YOLO trả về list, lấy phần đầu
        names = results_helmet.names  # danh sách class names
        detected = False

        for box in results_helmet.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id].lower()
            print("Helmet Detection:", cls_name)
            conf = float(box.conf[0])  # lấy độ tự tin
            print(f"🪖 Helmet Detection: {cls_name}, Confidence: {conf:.2%}")
            if "helmet" in cls_name and conf >= THRESH_HELMET_CONF:
                detected = True
                break

        if detected:
            results["helmet"] = "pass"
        else:
            results["helmet"] = "fail"
            box = test_boxes.get("helmet")
            if box:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                cv2.rectangle(test_image, (x1, y1), (x2, y2), colors["fail"], 2)
                cv2.putText(test_image, "helmet: fail", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["fail"], 2)
    else:
        results["helmet"] = "missing"
    face_crop = test_crops.get("face_smile")
    if face_crop is not None:
        smile_result = detect_smile(face_crop)
    else:
        smile_result = "missing"

    results["smile"] = smile_result
    # ==== 💾 OUTPUT ====

    output_path = os.path.join(OUTPUT_FOLDER, "test_result.jpg")
    cv2.imwrite(output_path, test_image)

    return output_path, results
