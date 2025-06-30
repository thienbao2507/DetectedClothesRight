import cv2
import numpy as np
import os

# ✅ Địa chỉ ảnh cần kiểm tra
TEST_IMAGE_PATH = "check/test/crop_nametag.jpg"

def check_nametag(image_path, bright_threshold=170, ratio_thresh=0.04, area_thresh=400, show=True):
    if not os.path.exists(image_path):
        print("❌ File không tồn tại:", image_path)
        return "missing"

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold để tìm vùng sáng (thẻ tên)
    _, binary = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)

    # 1. Tỷ lệ pixel sáng
    white_ratio = np.sum(binary == 255) / binary.size
    print(f"🔍 Bright pixel ratio: {white_ratio:.2%}")

    # 2. Tìm contour đủ lớn
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_thresh:
            found = True
            if show:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"Area: {int(area)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    if show:
        cv2.imshow("Original", img)
        cv2.imshow("Binary", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return "pass" if (white_ratio > ratio_thresh or found) else "fail"

# === CHẠY TRỰC TIẾP ===
if __name__ == "__main__":
    result = check_nametag(TEST_IMAGE_PATH)
    print(f"\n📋 Kết quả phát hiện bảng tên: {result}")
