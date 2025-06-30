import cv2
import numpy as np

def nothing(x):
    pass

# === Khởi tạo cửa sổ + thanh trượt
cv2.namedWindow("Detection")
cv2.resizeWindow("Detection", 500, 300)
cv2.createTrackbar("CLAHE clip", "Detection", 5, 40, nothing)      # 0.1 → 4.0 (x10)
cv2.createTrackbar("Blur k", "Detection", 3, 10, nothing)          # 3,5,7,...
cv2.createTrackbar("Canny T1", "Detection", 50, 200, nothing)
cv2.createTrackbar("Canny T2", "Detection", 150, 300, nothing)
cv2.createTrackbar("Area min", "Detection", 10, 1000, nothing)
cv2.createTrackbar("Area max", "Detection", 300, 2000, nothing)
cv2.createTrackbar("Circ min", "Detection", 50, 200, nothing)      # 0.5 → 2.0 (x100)
cv2.createTrackbar("Circ max", "Detection", 140, 300, nothing)

def detect_buttons_dynamic(image_path):
    while True:
        # === Đọc ảnh + resize nếu cần
        image = cv2.imread(image_path)
        if image is None:
            print("Ảnh không tồn tại")
            break
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # === Đọc giá trị từ trackbar
        clahe_clip = cv2.getTrackbarPos("CLAHE clip", "Detection") / 10.0
        blur_k = cv2.getTrackbarPos("Blur k", "Detection")
        blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1  # phải là số lẻ
        canny_t1 = cv2.getTrackbarPos("Canny T1", "Detection")
        canny_t2 = cv2.getTrackbarPos("Canny T2", "Detection")
        area_min = cv2.getTrackbarPos("Area min", "Detection")
        area_max = cv2.getTrackbarPos("Area max", "Detection")
        circ_min = cv2.getTrackbarPos("Circ min", "Detection") / 100
        circ_max = cv2.getTrackbarPos("Circ max", "Detection") / 100

        # === Tiền xử lý
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        blur = cv2.GaussianBlur(clahe_img, (blur_k, blur_k), 0)
        canny_edges = cv2.Canny(blur, canny_t1, canny_t2)

        # === Chia 2 nửa trái/phải để dò nút
        result_img = image.copy()
        pockets = [(0, 0, w // 2, h), (w // 2, 0, w // 2, h)]
        left_count, right_count = 0, 0

        for (x, y, ww, hh) in pockets:
            roi = clahe_img[y:y+hh, x:x+ww]
            roi_edges = cv2.Canny(roi, canny_t1, canny_t2)
            contours_roi, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours_roi:
                area = cv2.contourArea(cnt)
                if area_min < area < area_max:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circ_min < circularity < circ_max:
                        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
                        cx_center = x + x_cnt + w_cnt // 2
                        if cx_center < w // 2:
                            left_count += 1
                            cv2.rectangle(result_img, (x + x_cnt, y + y_cnt),
                                          (x + x_cnt + w_cnt, y + y_cnt + h_cnt), (0, 255, 0), 2)
                        else:
                            right_count += 1
                            cv2.rectangle(result_img, (x + x_cnt, y + y_cnt),
                                          (x + x_cnt + w_cnt, y + y_cnt + h_cnt), (0, 0, 255), 2)

        # === Ghi kết quả lên ảnh
        status = "✅ ĐỦ NÚT" if left_count >= 1 and right_count >= 1 else "❌ THIẾU NÚT"
        cv2.putText(result_img, f"{status} | Trái: {left_count} - Phải: {right_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # === Hiển thị kết quả
        cv2.imshow("Detection", result_img)

        # Thoát khi nhấn ESC
        key = cv2.waitKey(50)
        if key == 27:
            break

    cv2.destroyAllWindows()

# === Gọi hàm
detect_buttons_dynamic("check/test/crop_shirt.jpg")
