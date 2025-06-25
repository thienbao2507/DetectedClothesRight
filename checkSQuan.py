import cv2
import numpy as np

def detect_skin_box_in_pants(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không đọc được ảnh.")
        return

    # Xét phần 1/3 dưới ảnh
    height = image.shape[0]
    start_row = int(height * 2/3)
    lower_part = image[start_row:, :]

    # Chuyển sang HSV
    hsv = cv2.cvtColor(lower_part, cv2.COLOR_BGR2HSV)

    # Ngưỡng màu da HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Tính tỷ lệ da
    skin_ratio = np.sum(mask == 255) / mask.size
    print(f"🧪 Tỉ lệ da ở vùng chân: {skin_ratio:.2%}")

    result = image.copy()

    # Nếu có da → tìm contours và vẽ khung
    if skin_ratio > 0.04:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # bỏ vùng nhiễu nhỏ
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y + start_row), (x + w, y + h + start_row), (0, 0, 255), 2)
            cv2.putText(result, "pants_rolled_up", (x, y + start_row - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print("⚠️ Cảnh báo: Quần có thể bị sắn lên (da lộ ra).")
    else:
        print("✅ Không phát hiện da ở vùng ống quần.")

    # Hiển thị ảnh
    cv2.imshow("Original", image)
    cv2.imshow("Result (with box)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 📌 Gọi hàm
detect_skin_box_in_pants("check/test/crop_pants.jpg")
