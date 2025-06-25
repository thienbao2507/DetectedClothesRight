import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog

# Load image
img = cv2.imread('check/test/crop_shirt.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. CLAHE - tăng tương phản cục bộ
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)

# 2. LBP - đặc trưng texture
lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

# 3. HOG - đặc trưng cạnh và gradient
hog_features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, channel_axis=None)

# 4. Edge Detection - Canny
edges = cv2.Canny(gray, 100, 200)

# 5. Sobel Y (gradient theo chiều dọc)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# Hiển thị
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()

titles = ['Gốc (Grayscale)', 'CLAHE', 'LBP', 'HOG', 'Canny Edges', 'Sobel Y']
images = [gray, clahe_img, lbp, hog_image, edges, sobel_y_abs]

for i in range(6):
    axs[i].imshow(images[i], cmap='gray')
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()
