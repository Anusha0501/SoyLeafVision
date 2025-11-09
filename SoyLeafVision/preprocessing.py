import os
import cv2
from tqdm import tqdm
import albumentations as A

# Paths
RAW_DIR = "data/train/"
PROCESSED_DIR = "data/processed/"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Transformations
transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3)
])

for img_file in tqdm(os.listdir(RAW_DIR)):
    if img_file.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(RAW_DIR, img_file)
        img = cv2.imread(img_path)
        aug = transform(image=img)
        cv2.imwrite(os.path.join(PROCESSED_DIR, img_file), aug["image"])

print("✅ Preprocessing complete! Saved processed images to:", PROCESSED_DIR)
