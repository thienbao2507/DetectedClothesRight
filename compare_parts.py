import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Cấu hình
anchor_dir = "anchor"
test_dir = "test"
labels = ["nametag", "left_glove", "right_glove", "shirt", "pants",
          "left_shoe", "right_shoe", "left_arm", "right_arm"]

# Load model nhẹ
model = tf.keras.applications.MobileNetV2(
    include_top=False, weights='imagenet',
    input_shape=(224, 224, 3), pooling='avg'
)

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)

def compare(emb1, emb2, threshold=0.9):
    if emb1 is None or emb2 is None:
        return "missing"
    sim = cosine_similarity(emb1, emb2)[0][0]
    return "pass" if sim >= threshold else "fail"

# So sánh
results = {}
for label in labels:
    anchor_img = os.path.join(anchor_dir, f"crop_{label}.jpg")
    test_img = os.path.join(test_dir, f"crop_{label}.jpg")
    emb_anchor = extract_embedding(anchor_img)
    emb_test = extract_embedding(test_img)
    results[label] = compare(emb_anchor, emb_test)

# Hiển thị kết quả
import json
print(json.dumps(results, indent=2))
