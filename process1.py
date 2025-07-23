import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from scipy.signal import find_peaks
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

def detect_nametag_better(image_path, bright_threshold=170, ratio_thresh=0.04, area_thresh=400):
    img = cv2.imread(image_path)
    if img is None:
        return "missing", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)

    white_ratio = np.sum(binary == 255) / binary.size
    print(f"[Nametag] Bright pixel ratio: {white_ratio:.2%}")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_thresh:
            x, y, w, h = cv2.boundingRect(cnt)

            # Cắt vùng nghi ngờ là thẻ
            patch = gray[y:y + h, x:x + w]
            if patch.size == 0:
                continue

            # Tính độ tương phản và mức độ tối trung tâm
            center = patch[h // 4:3 * h // 4, w // 4:3 * w // 4]
            dark_ratio = np.sum(center < 100) / center.size
            contrast = np.std(patch)

            print(f"[Nametag] Area: {area}, Contrast: {contrast:.1f}, Dark center ratio: {dark_ratio:.2f}")

            if contrast > 20 and dark_ratio > 0.1:
                best_box = (x, y, x + w, y + h)
                break

    if best_box:
        return "pass", best_box
    return "fail", None

#===Áo===
def evaluate_shirt_color_hsv_direct(img, save_path=None):
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))

    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([3, 80, 80])
    upper_orange = np.array([25, 255, 255])

    # ROI giống như demo.py: vùng giữa ngực nơi hay có sọc cam
    top = int(h_img * 0.18)
    bottom = int(h_img * 0.42)
    left = int(w_img * 0.05)
    right = int(w_img * 0.95)
    roi = hsv[top:bottom, left:right]

    kernel = np.ones((5, 5), np.uint8)
    roi_mask = cv2.inRange(roi, lower_orange, upper_orange)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

    ref_top = np.array([38, 34, 247])
    ref_cam = np.array([3, 220, 231])
    ref_bottom = np.array([115, 141, 86])

    def is_similar_v2(color1, color2, h_thresh=25, s_thresh=130, v_thresh=130):
        h1, s1, v1 = color1
        h2, s2, v2 = color2
        return abs(h1 - h2) < h_thresh and abs(s1 - s2) < s_thresh and abs(v1 - v2) < v_thresh

    orange_range = (np.array([3, 40, 80]), np.array([30, 255, 255]))  # CAM linh hoạt hơn
    be_range = (np.array([5, 10, 100]), np.array([75, 80, 255]))
    blue_range = (np.array([95, 30, 35]), np.array([135, 255, 255]))  # XANH dương mở rộng

    def in_range(color, color_range):
        lower, upper = color_range
        return np.all(color >= lower) and np.all(color <= upper)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if save_path is not None:
            debug_img = img.copy()
            cv2.putText(debug_img, "No orange contour detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imwrite(save_path, debug_img)
        return "fail"

    largest_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_cnt) < 150:
        if save_path is not None:
            debug_img = img.copy()
            cv2.putText(debug_img, "Contour too small", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imwrite(save_path, debug_img)
        return "fail"

    x, y, w_box, h_box = cv2.boundingRect(largest_cnt)
    x_abs = x + left
    y_abs = y + top

    cam_roi = hsv[y_abs:y_abs + h_box, x_abs:x_abs + w_box]
    cam_mask = cv2.inRange(cam_roi, lower_orange, upper_orange)
    cam_mean = np.array(cv2.mean(cam_roi, mask=cam_mask)[:3])

    top_hsv = hsv[0:y_abs, x_abs:x_abs + w_box]
    top_mean = np.array(cv2.mean(top_hsv)[:3])

    bot_hsv = hsv[y_abs + h_box:, x_abs:x_abs + w_box]
    bot_mean = np.array(cv2.mean(bot_hsv)[:3])

    # cam_match = is_similar_v2(cam_mean, ref_cam, h_thresh=20, s_thresh=130, v_thresh=130)
    # top_match = is_similar_v2(top_mean, ref_top, h_thresh=25, s_thresh=130, v_thresh=130)
    # bottom_match = is_similar_v2(bot_mean, ref_bottom, h_thresh=25, s_thresh=130, v_thresh=130)
    cam_match = in_range(cam_mean, orange_range)
    top_match = in_range(top_mean, be_range)
    bottom_match = in_range(bot_mean, blue_range)

    result = "pass" if cam_match and top_match and bottom_match else "fail"

    if save_path is not None:
        debug_img = img.copy()

        # Vẽ vùng CAM
        cv2.rectangle(debug_img, (x_abs, y_abs), (x_abs + w_box, y_abs + h_box), (0, 165, 255), 2)
        cv2.putText(debug_img, "CAM", (x_abs, y_abs - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 165, 255), 1)

        # Vẽ vùng BE (top)
        cv2.rectangle(debug_img, (x_abs, 0), (x_abs + w_box, y_abs), (0, 255, 255), 2)
        cv2.putText(debug_img, "BE", (x_abs, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

        # Vẽ vùng XANH (bottom)
        cv2.rectangle(debug_img, (x_abs, y_abs + h_box), (x_abs + w_box, h_img), (255, 0, 0), 2)
        cv2.putText(debug_img, "BLUE", (x_abs, y_abs + h_box + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)

        # Nếu có vùng sai → ghi thông báo cụ thể
        if result == "fail":
            if not cam_match:
                cv2.putText(debug_img, "❌ Sai CAM", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not top_match:
                cv2.putText(debug_img, "❌ Sai BE (Tren)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not bottom_match:
                cv2.putText(debug_img, "❌ Sai BLUE (Duoi)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(debug_img, "✅ Dung mau dong phuc", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imwrite(save_path, debug_img)

    return result
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

    return crops, crop_paths, image
def extract_shirt_by_edge(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y).astype(np.uint8)

    row_mean = np.mean(sobel_y, axis=1)
    peaks, _ = find_peaks(row_mean, distance=30, prominence=20)
    if len(peaks) >= 2:
        peaks = sorted(peaks[:2])
        y1, y2 = peaks[0], peaks[1]
        top = img[0:y1, :, :]
        mid = img[y1:y2, :, :]
        bot = img[y2:, :, :]

        return {
            "top": np.mean(top.reshape(-1, 3), axis=0),
            "mid": np.mean(mid.reshape(-1, 3), axis=0),
            "bot": np.mean(bot.reshape(-1, 3), axis=0)
        }
    return None

# def extract_shirt_colors(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         return None
#     img = cv2.resize(img, (300, 300))  # đảm bảo tỷ lệ chuẩn
#
#     h = img.shape[0]
#     top = img[0:int(h/3), :, :]
#     mid = img[int(h/3):int(2*h/3), :, :]
#     bot = img[int(2*h/3):, :, :]
#
#     color_top = np.mean(top.reshape(-1, 3), axis=0)
#     color_mid = np.mean(mid.reshape(-1, 3), axis=0)
#     color_bot = np.mean(bot.reshape(-1, 3), axis=0)
#
#     return {
#         "top": color_top,
#         "mid": color_mid,
#         "bot": color_bot
#     }

# ==DeLoy==
def run_inference(test_image_path):
    # Tạo lại thư mục kết quả
    os.makedirs(f"{OUTPUT_FOLDER}/anchor", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/test", exist_ok=True)
    box_errors = []
#check
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
        if label == "nametag":
            if early_fail:
                results[label] = "fail"
                continue
            result, nametag_box = detect_nametag_better(test_paths.get(label))
            offset = test_boxes["nametag"]
            if nametag_box:
                x1_crop, y1_crop, x2_crop, y2_crop = nametag_box
                x1 = offset["x1"] + x1_crop
                y1 = offset["y1"] + y1_crop
                x2 = offset["x1"] + x2_crop
                y2 = offset["y1"] + y2_crop
        else:
            if label == "shirt":
                shirt_path = test_paths.get("shirt")
                shirt_img = cv2.imread(shirt_path)

                result = "missing"
                if shirt_img is not None:
                    debug_path = os.path.join(OUTPUT_FOLDER, "test", "shirt_debug.jpg")
                    result = evaluate_shirt_color_hsv_direct(shirt_img, save_path=debug_path)
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
                        # result = "fail"
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
                        box = test_boxes[arm_label]
                        cv2.rectangle(test_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), colors["fail"], 2)
                        cv2.putText(test_image, f"{arm_label}_skin", (box["x1"], box["y1"] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["fail"], 2)

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

    if box_errors:
        for err in box_errors:
            x1, y1, x2, y2 = err["box"]
            color = err["color"]
            cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(test_image, f"{err['label']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # ==== 💾 OUTPUT ====
    output_path = os.path.join(OUTPUT_FOLDER, "test_result.jpg")
    cv2.imwrite(output_path, test_image)

    return output_path, results
