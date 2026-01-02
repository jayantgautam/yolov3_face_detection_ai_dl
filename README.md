# YOLOv3 Face Detection (PyTorch + OpenCV) (by Jayant Gautam)

This project implements a complete YOLOv3-based face detection pipeline, covering data preparation, model training, ONNX export, and real-time inference using OpenCV.

# Feature

Custom YOLOv3 implementation in PyTorch

Automated dataset creation and augmentation with OpenCV

IoU-based k-means anchor box optimization for faces

Mixed-precision training for efficiency

ONNX export for cross-platform deployment

Real-time inference using OpenCV DNN

# Project Structure
├── Open_CV_annotations_for_YOLO.py   # Dataset creation & labeling

├── Calc_Anchors_kMeans.py            # Anchor box calculation

├── YOLO_v3.py                        # YOLOv3 model & training

├── YOLO_v3_ONNX_exp.py               # ONNX export

├── detect_img.py                     # Inference with OpenCV

# Usage

Train the model

python YOLO_v3.py


Run inference

python detect_img.py --image test.jpg

# Results

~78% mAP on face detection

~42 FPS inference speed

Lightweight input size (160×160) optimized for real-time use

# Tech Stack

Python . PyTorch · OpenCV · NumPy · ONNX
