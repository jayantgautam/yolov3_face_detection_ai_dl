import numpy as np

def iou_wh(box, clusters):
    
    """
    Calculate the IoU between a box and cluster anchors.
    box: numpy array of shape (2,) [w, h].dkejebdnen
    clusters: numpy array of shape (k, 2)
    """
    
    w, h = box
    cluster_w = clusters[:, 0]
    cluster_h = clusters[:, 1]

    inter_w = np.minimum(w, cluster_w)
    inter_h = np.minimum(h, cluster_h)
    inter_area = inter_w * inter_h

    box_area = w * h
    cluster_area = cluster_w * cluster_h

    iou = inter_area / (box_area + cluster_area - inter_area + 1e-10)
    return iou

def kmeans_anchors(boxes, k=9, dist_threshold=1e-6, max_iter=1000):
    
    """
    Run k-means clustering with IoU-based distance on box widths and heights.
    boxes: numpy array of shape (N, 2) where each row is (w, h) in pixels
    k: number of clusters (anchors)
    """
    
    # Initialize clusters by randomly picking k boxes
    clusters = boxes[np.random.choice(boxes.shape[0], k, replace=False)]

    for iteration in range(max_iter):
        distances = []
        for box in boxes:
            ious = iou_wh(box, clusters)
            distances.append(1 - ious)  # distance = 1 - IoU
        distances = np.array(distances)  # shape (N, k)

        nearest_clusters = np.argmin(distances, axis=1)

        new_clusters = []
        for cluster_idx in range(k):
            cluster_boxes = boxes[nearest_clusters == cluster_idx]
            if len(cluster_boxes) == 0:
                # No boxes assigned to this cluster, reinitialize randomly
                new_clusters.append(clusters[cluster_idx])
            else:
                # Update cluster to be median width and height
                median_w = np.median(cluster_boxes[:, 0])
                median_h = np.median(cluster_boxes[:, 1])
                new_clusters.append([median_w, median_h])
        new_clusters = np.array(new_clusters)

        # Check convergence
        diff = np.abs(new_clusters - clusters).sum()
        if diff < dist_threshold:
            break
        clusters = new_clusters

    return clusters


import glob

def load_all_boxes(label_dir, image_size=160):
    boxes = []
    for file in glob.glob(f"{label_dir}/*.txt"):
        with open(file) as f:
            for line in f.readlines():
                class_id, x, y, w, h = map(float, line.strip().split())
                boxes.append([w * image_size, h * image_size])  # convert normalized to pixels
    return np.array(boxes)

boxes = load_all_boxes('data/faces/train/labels', image_size=160)
anchors_pixel = kmeans_anchors(boxes, k=9)

anchors_normalized = anchors_pixel / 160.0  # Normalize back to 0-1 scale

# Group into 3 scales (YOLOv3 style)
anchors = anchors_normalized.reshape(3, 3, 2).tolist()

print("New anchors per scale:")
for i, scale in enumerate(anchors):
    print(f"Scale {i+1}:", scale)
