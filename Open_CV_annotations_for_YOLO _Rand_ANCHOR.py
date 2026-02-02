import cv2
import os
import random
from os import path, walk

# Set base path
base_dir = path.dirname(path.abspath(__file__))  # Note: now resolves correctly

# Subfolder with images
subfolder = r"data_for_image_proc"
data_dir = path.join(base_dir, subfolder)

# Output directories
annotation_dir = path.join(base_dir, r"annotations")
os.makedirs(annotation_dir, exist_ok=True)

image_dir = path.join(base_dir, r"images")
os.makedirs(image_dir, exist_ok=True)

# Get filenames
filenames = next(walk(data_dir), (None, None, []))[2]

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(r"E:\AI project\haarcascade_frontalface_default.xml")

image_size = 160  # Final training size
max_shift = 0.2  # Max shift as a fraction of crop size

for filename in filenames:
    img_path = path.join(data_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read image {filename}")
        continue

    height, width = img.shape[:2]
    crop_size = min(height, width)

    # Apply random shift before cropping
    shift_x = int(random.uniform(-max_shift, max_shift) * crop_size)
    shift_y = int(random.uniform(-max_shift, max_shift) * crop_size)

    center_x = width // 2 + shift_x
    center_y = height // 2 + shift_y

    # Clamp center to stay within bounds
    center_x = max(crop_size // 2, min(width - crop_size // 2, center_x))
    center_y = max(crop_size // 2, min(height - crop_size // 2, center_y))

    # Crop around new center
    start_x = center_x - crop_size // 2
    start_y = center_y - crop_size // 2
    cropped_img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]

    resized_img = cv2.resize(cropped_img, (image_size, image_size))
    output_image_path = path.join(image_dir, filename)
    cv2.imwrite(output_image_path, resized_img)

    # Detect face on cropped images
    face_rects = face_cascade.detectMultiScale(
        cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=4
    )

    annotation_file = path.join(annotation_dir, filename.split('.')[0] + '.txt')
    with open(annotation_file, 'w') as ann_file:
        for (x, y, w, h) in face_rects:
            x_center = (x + w / 2) / crop_size
            y_center = (y + h / 2) / crop_size
            width_norm = w / crop_size
            height_norm = h / crop_size
            ann_file.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

print("Random shifted face data generated.")

