import cv2
import numpy as np
import os

# ✅ Địa chỉ ảnh cần kiểm tra
TEST_IMAGE_PATH = "check/test/crop_nametag.jpg"

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
# === CHẠY TRỰC TIẾP ===
if __name__ == "__main__":
    result = check_nametag(TEST_IMAGE_PATH)
    print(f"\n📋 Kết quả phát hiện bảng tên: {result}")
