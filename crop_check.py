import cv2
import mediapipe as mp
import json
import os

# Cấu hình
IMAGE_PATH = "test/anchor_000.jpg"
OUTPUT_FOLDER = "check"
RESIZED_SHAPE = (512, 1024)
OUTPUT_JSON = f"{OUTPUT_FOLDER}/crop_coordinates.json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load ảnh
image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, RESIZED_SHAPE)
h, w, _ = image.shape

# Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if not results.pose_landmarks:
    print("❌ Không phát hiện được người.")
    exit()

landmarks = results.pose_landmarks.landmark

def get_point(landmark):
    return int(landmark.x * w), int(landmark.y * h)

def save_crop(label, x1, y1, x2, y2):
    if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(f"{OUTPUT_FOLDER}/crop_{label}.jpg", crop)
        crops[label] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

# Màu sắc từng vùng
colors = {
    "nametag": (0, 255, 0),
    "shirt": (0, 255, 255),
    "pants": (255, 255, 0),
    "left_glove": (255, 0, 0),
    "right_glove": (255, 0, 0),
    "left_shoe": (128, 0, 255),
    "right_shoe": (128, 0, 255),
    "left_arm": (0, 128, 255),
    "right_arm": (0, 128, 255),
}

crops = {}

# Các điểm cơ bản
ls, rs = landmarks[11], landmarks[12]
lw, rw = landmarks[15], landmarks[16]
le, re = landmarks[13], landmarks[14]
la, ra = landmarks[27], landmarks[28]
lh, rh = landmarks[23], landmarks[24]

# 1. Nametag (giữa 2 vai, dịch xuống)
x1, y1 = get_point(ls)
x2, y2 = get_point(rs)

# Lấy phần nửa phải (từ giữa vai đến vai phải)
cx1 = int((x1 + x2) *(1/2))       # lệch phải
cx2 = max(x1, x2) +20   # mở rộng thêm nếu muốn

cy1 = int((y1 + y2) / 2 + 0.08 * h)
cy2 = cy1 + 100

save_crop("nametag", cx1, cy1, cx2, cy2)


# 2. Găng tay (dùng các điểm tay)
def crop_hand(label, ids):
    pts = [landmarks[i] for i in ids]
    xs = [int(p.x * w) for p in pts]
    ys = [int(p.y * h) for p in pts]
    # Dùng margin lớn hơn cho vùng găng tay
    margin_x = 30
    margin_y = 50
    x1, x2 = min(xs) - margin_x, max(xs) + margin_x
    y1, y2 = min(ys) - margin_y, max(ys) + margin_y

    save_crop(label, x1, y1, x2, y2)

crop_hand("left_glove", [15, 17, 19, 21])
crop_hand("right_glove", [16, 18, 20, 22])

# 3. Giày (tăng chiều cao)
for label, point in zip(["left_shoe", "right_shoe"], [la, ra]):
    px, py = get_point(point)
    sx1, sy1 = px - 50, py - 20
    sx2, sy2 = px + 50, py + 60
    save_crop(label, sx1, sy1, sx2, sy2)

# 4. Áo
x_ls, y_ls = get_point(ls)
x_rs, y_rs = get_point(rs)

shirt_x1 = min(x_ls, x_rs) - 20
shirt_y1 = min(y_ls, y_rs) - 10
shirt_x2 = max(x_ls, x_rs) + 20

shirt_y2 = int((lh.y + rh.y) / 2 * h)
save_crop("shirt", shirt_x1, shirt_y1, shirt_x2, shirt_y2)

# 5. Quần
lx, ly = get_point(lh)
rx, ry = get_point(rh)
ankle_y = max(get_point(la)[1], get_point(ra)[1])
pants_x1 = min(lx, rx) - 80
pants_x2 = max(lx, rx) + 80
pants_y1 = int(min(ly, ry))
pants_y2 = ankle_y + 40
save_crop("pants", pants_x1, pants_y1, pants_x2, pants_y2)

# 6. Cánh tay (shoulder → wrist)
for label, shoulder, wrist in zip(["left_arm", "right_arm"], [ls, rs], [lw, rw]):
    sx, sy = get_point(shoulder)
    wx, wy = get_point(wrist)
    ax1, ay1 = min(sx, wx) - 30, min(sy, wy) - 30
    ax2, ay2 = max(sx, wx) + 30, max(sy, wy) + 30
    save_crop(label, ax1, ay1, ax2, ay2)

# Vẽ box
for label, box in crops.items():
    color = colors.get(label, (0, 255, 0))
    cv2.rectangle(image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
    cv2.putText(image, label, (box["x1"], box["y1"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Lưu ảnh & toạ độ
cv2.imwrite(f"{OUTPUT_FOLDER}/visual_check.jpg", image)
with open(OUTPUT_JSON, "w") as f:
    json.dump(crops, f, indent=2)

print("✅ Đã crop chính xác bàn tay và lưu vào thư mục:", OUTPUT_FOLDER)
