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

# Load mô hình helmet
helmet_model = YOLO("best.pt")
helmet_model.eval()

# ==== ⚙️ CONFIG ====
ANCHOR_IMAGE_PATH = "anchors_cropped/anchor_019.jpg"
OUTPUT_FOLDER = "check"
RESIZED_SHAPE = (512, 1024)
THRESHOLD = 0.75

# ==== 📂 INIT ====
os.makedirs(f"{OUTPUT_FOLDER}/anchor", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/test", exist_ok=True)

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
def check_nametag(image_path, bright_threshold=170, ratio_thresh=0.03, area_thresh=300, show=True):
    if not os.path.exists(image_path):
        print("❌ File không tồn tại:", image_path)
        return "missing", None

    img = cv2.imread(image_path)
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
            best_box = (x, y, x+w, y+h)
            found = area > area_thresh

    # Tính tỷ lệ pixel sáng dựa trên contour lớn nhất
    if largest_area > 0:
        white_ratio = largest_area / binary.size
        print(f"🔍 Bright pixel ratio (largest cluster): {white_ratio:.2%}")

    if show and found:
        cv2.rectangle(img, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
        cv2.putText(img, f"Area: {int(largest_area)}", (best_box[0], best_box[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    if show:
        cv2.imwrite("output_nametag_result.jpg", img)
        cv2.imwrite("output_nametag_binary.jpg", binary)
        print("🖼️ Ảnh kết quả đã được lưu: output_nametag_result.jpg & output_nametag_binary.jpg")

    return ("pass" if (white_ratio > ratio_thresh or found) else "fail"), best_box
# ==== 📌 POSE CROP ====
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
    cy1 = int((y1 + y2) * 0.5 + 0.08 * h)-20
    cy2 = cy1 + 100 #độ dài
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

    # === CROP VÙNG ĐẦU (HELMET) ===
    head_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # mũi, mắt, miệng, tai
    head_points = [get_point(landmarks[i]) for i in head_ids]
    xs = [p[0] for p in head_points]
    ys = [p[1] for p in head_points]

    # Mở rộng vùng đầu để lấy cả nón
    margin_x, margin_y = 40, 60
    x1, y1 = max(min(xs) - margin_x, 0), max(min(ys) - margin_y, 0)
    x2, y2 = min(max(xs) + margin_x, w), min(max(ys) + margin_y, h)

    save_crop("helmet", x1, y1, x2, y2)

    return crops, crop_paths, image

def extract_shirt_colors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (300, 300))  # đảm bảo tỷ lệ chuẩn

    h = img.shape[0]
    top = img[0:int(h/3), :, :]
    mid = img[int(h/3):int(2*h/3), :, :]
    bot = img[int(2*h/3):, :, :]

    color_top = np.mean(top.reshape(-1, 3), axis=0)
    color_mid = np.mean(mid.reshape(-1, 3), axis=0)
    color_bot = np.mean(bot.reshape(-1, 3), axis=0)

    return {
        "top": color_top,
        "mid": color_mid,
        "bot": color_bot
    }

# ==DeLoy==
def run_inference(test_image_path):
    # Tạo lại thư mục kết quả
    os.makedirs(f"{OUTPUT_FOLDER}/anchor", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/test", exist_ok=True)
    box_errors = []

    # ==== 📌 POSE CROP ====
    print("🔧 Đang crop ảnh chuẩn...")
    anchor_boxes, anchor_paths, _ = crop_pose(ANCHOR_IMAGE_PATH, f"{OUTPUT_FOLDER}/anchor")

    print("🔧 Đang crop ảnh test...")
    test_boxes, test_paths, test_image = crop_pose(test_image_path, f"{OUTPUT_FOLDER}/test")

    results = {}
    early_fail = False

    for label in labels:
        if label in ["left_arm", "right_arm"]:
            continue  # bỏ kiểm tra tay áo ở bước này, sẽ kiểm tra sau nếu shirt pass
        if label in ["left_glove", "right_glove"]:
            path = test_paths.get(label)
            if path is None:
                result = "missing"
            else:
                img = cv2.imread(path)
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

                    # === Tổng hợp kết luận
                    if skin_ratio_full > 0.4:
                        result = "fail"
                        if label in test_boxes:
                            box = test_boxes[label]
                            box_errors.append({
                                "label": f"{label}_no_glove",
                                "box": (box["x1"], box["y1"], box["x2"], box["y2"]),
                                "color": (0, 0, 255)
                            })
                    elif skin_ratio_tip > 0.02:
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
            result, nametag_box = check_nametag(test_paths.get(label))
            offset = test_boxes.get("nametag")
            if nametag_box and offset:
                x1_crop, y1_crop, x2_crop, y2_crop = nametag_box
                x1 = offset["x1"] + x1_crop
                y1 = offset["y1"] + y1_crop
                x2 = offset["x1"] + x2_crop
                y2 = offset["y1"] + y2_crop
            results[label] = result
            
        else:
            if label == "shirt":
                anchor_colors = extract_shirt_colors(anchor_paths.get(label))
                test_colors = extract_shirt_colors(test_paths.get(label))
                result = "missing"
                if anchor_colors and test_colors:
                    diff_top = np.linalg.norm(anchor_colors["top"] - test_colors["top"])
                    diff_mid = np.linalg.norm(anchor_colors["mid"] - test_colors["mid"])
                    diff_bot = np.linalg.norm(anchor_colors["bot"] - test_colors["bot"])

                    print(f"[SHIRT COLOR DIFF] Top: {diff_top:.1f}, Mid: {diff_mid:.1f}, Bot: {diff_bot:.1f}")

                    if diff_top < 40 and diff_mid < 40 and diff_bot < 40:
                        result = "pass"
                    else:
                        result = "fail"

            if label in ["shirt", "pants"] and result == "fail":
                early_fail = True
                # Nếu pants là pass, kiểm tra xem có bị sắn (lộ da) không
            if label == "pants" and result == "pass":
                path = test_paths.get("pants")
                if path is not None:
                    img = cv2.imread(path)
                    h = img.shape[0]
                    start_row = int(h * 2 / 3)
                    lower_part = img[start_row:, :]

                    hsv = cv2.cvtColor(lower_part, cv2.COLOR_BGR2HSV)
                    lower = np.array([0, 20, 70], dtype=np.uint8)
                    upper = np.array([20, 255, 255], dtype=np.uint8)
                    mask = cv2.inRange(hsv, lower, upper)

                    skin_ratio = np.sum(mask == 255) / mask.size
                    print(f"[PANTS SẮN] Skin ratio (lower): {skin_ratio:.2%}")

                    if skin_ratio > 0.04:
                        result = "fail"
                        # ✅ Tìm contour để vẽ box da vùng ống quần dưới
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 100:
                                continue
                            x, y, w, h = cv2.boundingRect(cnt)
                            if "pants" in test_boxes:
                                box = test_boxes["pants"]
                                x1 = box["x1"] + x
                                y1 = box["y1"] + start_row + y
                                x2 = x1 + w
                                y2 = y1 + h
                                box_errors.append({
                                    "label": "pants_rolled_up",
                                    "box": (x1, y1, x2, y2),
                                    "color": (0, 0, 255)
                                })

                results["pants"] = result
        if label == "shirt" and result == "pass":
            for arm_label in ["left_arm", "right_arm"]:
                path = test_paths.get(arm_label)
                if path is None:
                    results[arm_label] = "missing"
                    continue

                img = cv2.imread(path)

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

                if skin_ratio > 0.04:
                    results[arm_label] = "fail"
                    if arm_label in test_boxes:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 100:
                                continue
                            x, y, w, h = cv2.boundingRect(cnt)
                            box = test_boxes[arm_label]
                            x1 = box["x1"] + x
                            y1 = box["y1"] + int(h / 3) + y
                            x2 = x1 + w
                            y2 = y1 + h
                            box_errors.append({
                                "label": f"{arm_label}_skin",
                                "box": (x1, y1, x2, y2),
                                "color": (0, 0, 255)
                            })
                else:
                    results[arm_label] = "pass"
        results[label] = result

        # 🎨 Vẽ khung lên ảnh
        if result == "fail":
            color = colors["fail"]
            if label == "nametag" and nametag_box:
                cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(test_image, f"{label}: {result}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif label in test_boxes:
                box = test_boxes[label]
                cv2.rectangle(test_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
                cv2.putText(test_image, f"{label}: {result}", (box["x1"], box["y1"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # ==== HELMET CHECK ====
    helmet_path = test_paths.get("helmet")
    if helmet_path is not None:
        results_helmet = helmet_model(helmet_path)[0]  # YOLO trả về list, lấy phần đầu
        names = results_helmet.names  # danh sách class names
        detected = False

        for box in results_helmet.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id].lower()
            print("Helmet Detection:", cls_name)
            if "helmet" in cls_name:
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

    # ==== 💾 OUTPUT ====
    output_path = os.path.join(OUTPUT_FOLDER, "test_result.jpg")
    cv2.imwrite(output_path, test_image)

    return output_path, results