import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolov10_chatGPT2b import YOLOv10
import torchvision.ops as ops  # For NMS
import os
import random
from sklearn.cluster import AgglomerativeClustering

from credentials import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG
MODEL_WEIGHTS = 'best_yolov10_pc.pth'


def get_random_test_image():
    test_dir = DATA_CONFIG['TEST_IMG_DIR']
    if not os.path.exists(test_dir):
        test_dir = DATA_CONFIG['VAL_IMG_DIR']
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise FileNotFoundError("No image files found in test directory")

    random_image = random.choice(image_files)
    return os.path.join(test_dir, random_image)


# ---------- Non-Max Suppression ----------
# def non_max_suppression(predictions, conf_threshold=0.5, nms_threshold=0.3):
#     boxes, confidences, class_ids = [], [], []

#     for pred in predictions[0]:  # batch size = 1
#         class_scores = pred[5:]
#         class_id = torch.argmax(class_scores).item()
#         class_prob = class_scores[class_id]
#         conf = pred[4]
#         score = conf * class_prob

#         # print(f"conf: {conf:.2f}, class_prob: {class_prob:.4f}, score: {score:.4f}")
#         if score < conf_threshold:
#             continue

#         cx, cy, w, h = pred[0:4]
#         x1 = cx - w / 2
#         y1 = cy - h / 2
#         x2 = cx + w / 2
#         y2 = cy + h / 2

#         boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
#         confidences.append(score.item())
#         class_ids.append(class_id)

#     if not boxes:
#         return [], [], []

#     boxes = torch.tensor(boxes)
#     confidences = torch.tensor(confidences)
#     class_ids = torch.tensor(class_ids)

#     keep = ops.nms(boxes, confidences, nms_threshold)
#     return boxes[keep], confidences[keep], class_ids[keep]


# def non_max_suppression(predictions, conf_threshold=0.5, nms_threshold=0.3, merge_class_boxes=True):
#     boxes, confidences, class_ids = [], [], []

#     for pred in predictions[0]:  # batch size = 1
#         class_scores = pred[5:]
#         class_id = torch.argmax(class_scores).item()
#         class_prob = class_scores[class_id]
#         conf = pred[4]
#         score = conf * class_prob

#         if score < conf_threshold:
#             continue

#         cx, cy, w, h = pred[0:4]
#         x1 = cx - w / 2
#         y1 = cy - h / 2
#         x2 = cx + w / 2
#         y2 = cy + h / 2

#         boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
#         confidences.append(score.item())
#         class_ids.append(class_id)

#     if not boxes:
#         return [], [], []

#     boxes = torch.tensor(boxes)
#     confidences = torch.tensor(confidences)
#     class_ids = torch.tensor(class_ids)

#     final_boxes, final_confidences, final_class_ids = [], [], []

#     for cls in class_ids.unique():
#         cls_mask = class_ids == cls
#         cls_boxes = boxes[cls_mask]
#         cls_confidences = confidences[cls_mask]

#         keep = ops.nms(cls_boxes, cls_confidences, nms_threshold)
#         kept_boxes = cls_boxes[keep]
#         kept_confidences = cls_confidences[keep]

#         if merge_class_boxes and kept_boxes.size(0) > 0:
#             merged = merge_boxes(kept_boxes)
#             final_boxes.append(merged.unsqueeze(0))
#             final_confidences.append(torch.tensor([kept_confidences.max()]))
#             final_class_ids.append(torch.tensor([cls]))
#         else:
#             final_boxes.append(kept_boxes)
#             final_confidences.append(kept_confidences)
#             final_class_ids.append(torch.full_like(kept_confidences, cls))

#     return (
#         torch.cat(final_boxes) if final_boxes else [],
#         torch.cat(final_confidences) if final_confidences else [],
#         torch.cat(final_class_ids) if final_class_ids else [],
#     )
# def merge_boxes(boxes):
#     # Merge multiple boxes into one by taking the min/max corners
#     x1 = boxes[:, 0].min()
#     y1 = boxes[:, 1].min()
#     x2 = boxes[:, 2].max()
#     y2 = boxes[:, 3].max()
#     return torch.tensor([x1, y1, x2, y2])


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def merge_grouped_boxes(boxes):
    x1 = boxes[:, 0].min()
    y1 = boxes[:, 1].min()
    x2 = boxes[:, 2].max()
    y2 = boxes[:, 3].max()
    return torch.tensor([x1, y1, x2, y2])


def group_boxes_by_iou(boxes, iou_threshold=0.4):
    groups = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            if iou(boxes[i], boxes[j]) > iou_threshold:
                group.append(j)
                used.add(j)
        groups.append(group)
    return groups


def non_max_suppression(predictions, conf_threshold=0.5, nms_threshold=0.4):
    boxes, confidences, class_ids = [], [], []

    for pred in predictions[0]:  # batch size = 1
        class_scores = pred[5:]
        class_id = torch.argmax(class_scores).item()
        class_prob = class_scores[class_id]
        conf = pred[4]
        score = conf * class_prob

        if score < conf_threshold:
            continue

        cx, cy, w, h = pred[0:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
        confidences.append(score.item())
        class_ids.append(class_id)

    if not boxes:
        return [], [], []

    boxes = torch.tensor(boxes)
    scores = torch.tensor(confidences)
    class_ids = torch.tensor(class_ids)

    # Group boxes by class and merge overlapping ones
    final_boxes = []
    final_scores = []
    final_class_ids = []

    unique_classes = class_ids.unique()

    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep = ops.nms(cls_boxes, cls_scores, nms_threshold)
        kept_boxes = cls_boxes[keep]
        kept_scores = cls_scores[keep]

        # Group kept boxes into one merged box per object
        for i in range(len(keep)):
            ious = ops.box_iou(kept_boxes[i].unsqueeze(0), cls_boxes)[0]
            group_mask = ious > 0.4  # merge boxes overlapping enough

            merged_box = torch.cat([
                cls_boxes[group_mask][:, [0, 1]].min(dim=0).values,
                cls_boxes[group_mask][:, [2, 3]].max(dim=0).values
            ])
            final_boxes.append(merged_box)
            final_scores.append(kept_scores[i])
            final_class_ids.append(cls.item())

    return torch.stack(final_boxes), torch.tensor(final_scores), torch.tensor(final_class_ids)

def merge_overlapping_boxes(boxes, scores, class_ids, iou_threshold=0.5):
    merged_boxes = []
    merged_scores = []
    merged_class_ids = []

    for cls in class_ids.unique():
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        used = torch.zeros(len(cls_boxes), dtype=torch.bool)

        for i in range(len(cls_boxes)):
            if used[i]:
                continue

            current_box = cls_boxes[i]
            overlaps = ops.box_iou(current_box.unsqueeze(0), cls_boxes)[0]
            cluster_mask = overlaps > iou_threshold

            cluster_boxes = cls_boxes[cluster_mask]
            cluster_scores = cls_scores[cluster_mask]

            # Merge cluster to one box
            x1 = cluster_boxes[:, 0].min()
            y1 = cluster_boxes[:, 1].min()
            x2 = cluster_boxes[:, 2].max()
            y2 = cluster_boxes[:, 3].max()

            merged_boxes.append(torch.tensor([x1, y1, x2, y2]))
            merged_scores.append(cluster_scores.mean())
            merged_class_ids.append(cls.item())

            used |= cluster_mask

    return torch.stack(merged_boxes), torch.tensor(merged_scores), torch.tensor(merged_class_ids)

def cluster_and_merge_boxes(boxes, scores, class_ids, distance_thresh=20):
    merged_boxes = []
    merged_scores = []
    merged_class_ids = []

    for cls in class_ids.unique():
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Get box centers
        centers = ((cls_boxes[:, 0] + cls_boxes[:, 2]) / 2,
                   (cls_boxes[:, 1] + cls_boxes[:, 3]) / 2)
        centers_np = torch.stack(centers, dim=1).numpy()

        if len(centers_np) <= 1:
            # Only one box — keep as is
            merged_boxes.append(cls_boxes[0])
            merged_scores.append(cls_scores[0])
            merged_class_ids.append(cls.item())
            continue

        # Cluster centers
        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=distance_thresh).fit(centers_np)
        labels = clustering.labels_

        for lbl in np.unique(labels):
            cluster_mask = labels == lbl
            cluster_boxes = cls_boxes[cluster_mask]
            cluster_scores = cls_scores[cluster_mask]

            x1 = cluster_boxes[:, 0].min()
            y1 = cluster_boxes[:, 1].min()
            x2 = cluster_boxes[:, 2].max()
            y2 = cluster_boxes[:, 3].max()

            merged_boxes.append(torch.tensor([x1, y1, x2, y2]))
            merged_scores.append(cluster_scores.mean())
            merged_class_ids.append(cls.item())

    return torch.stack(merged_boxes), torch.tensor(merged_scores), torch.tensor(merged_class_ids)




def decode_output(output, stride, num_classes):
    B, C, H, W = output.shape
    output = output.view(B, 3, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()

    cx = torch.sigmoid(output[..., 0])
    cy = torch.sigmoid(output[..., 1])
    w = output[..., 2]
    h = output[..., 3]
    obj_conf = torch.sigmoid(output[..., 4])
    class_scores = torch.sigmoid(output[..., 5:])

    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_x = grid_x.to(output.device).float()
    grid_y = grid_y.to(output.device).float()

    cx = (cx + grid_x.unsqueeze(0).unsqueeze(0)) * stride
    cy = (cy + grid_y.unsqueeze(0).unsqueeze(0)) * stride
    w = torch.exp(w) * stride
    h = torch.exp(h) * stride

    boxes = torch.stack([cx, cy, w, h, obj_conf], dim=-1)
    final = torch.cat([boxes, class_scores], dim=-1)
    return final.view(B, -1, 5 + num_classes)


# ---------- Detection Pipeline ----------
def detect():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv10(variant=TRAIN_CONFIG['MODEL_VARIANT'], num_classes=MODEL_CONFIG['NUM_CLASSES']).to(device)

    checkpoint = torch.load(MODEL_WEIGHTS)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device).keys('model_state_dict'))
    model.eval()

    img_path = get_random_test_image()
    img = cv2.imread(img_path)
    resized = cv2.resize(img, (640, 640))

    resized_rgb = resized[:, :, ::-1].copy()
    input_tensor = torch.from_numpy(resized_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        out_s, out_m, out_l = model(input_tensor)

    print(f"Small head output shape: {out_s.shape}")
    print(f"Medium head output shape: {out_m.shape}")
    print(f"Large head output shape: {out_l.shape}")

    decoded_s = decode_output(out_s, stride=8, num_classes=MODEL_CONFIG['NUM_CLASSES'])
    decoded_m = decode_output(out_m, stride=16, num_classes=MODEL_CONFIG['NUM_CLASSES'])
    decoded_l = decode_output(out_l, stride=32, num_classes=MODEL_CONFIG['NUM_CLASSES'])
    predictions = torch.cat([decoded_s, decoded_m, decoded_l], dim=1)
    print("Final prediction shape:", predictions.shape)

    threshold = 0.3
    confidences_raw = predictions[0][:, 4]
    above_thresh = confidences_raw > threshold
    print(f"\nTotal raw predictions: {predictions.shape[1]}")
    print(f"Predictions above threshold ({threshold}): {above_thresh.sum().item()}")
    print(f"Prediction : {confidences_raw}")

    # Existing post-processing
    boxes, confidences, class_ids = non_max_suppression(predictions, conf_threshold=threshold, nms_threshold=0.4)
    print(f"Initial NMS box count: {len(boxes)}")

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = MODEL_CONFIG['CLASS_NAMES'][class_id]
        cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized, f'{class_id:.0f}', (x1, y1 - 10), #{class_name} {conf:.2f}
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Merge clusters of overlapping boxes into single objects
    # boxes, confidences, class_ids = merge_overlapping_boxes(boxes, confidences, class_ids, iou_threshold=0.5)
    # print(f"Final merged object box count: {len(boxes)}")

    distance_thresh = 50
    while len(boxes) > 20:
        boxes, confidences, class_ids = cluster_and_merge_boxes(boxes, confidences, class_ids, distance_thresh=distance_thresh)
        print(f"Final merged object box count: {len(boxes)}")
        distance_thresh += 50


    # boxes, confidences, class_ids = cluster_and_merge_boxes(boxes, confidences, class_ids, distance_thresh=50)
    # print(f"Final merged object box count: {len(boxes)}")
    # boxes, confidences, class_ids = cluster_and_merge_boxes(boxes, confidences, class_ids, distance_thresh=100)
    # print(f"Final merged object box count: {len(boxes)}")
    # boxes, confidences, class_ids = cluster_and_merge_boxes(boxes, confidences, class_ids, distance_thresh=150)
    # print(f"Final merged object box count: {len(boxes)}")

    print("NMS Results:")
    print("Boxes:", boxes)
    print("Confidences:", confidences)
    print("Class IDs:", class_ids)

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = MODEL_CONFIG['CLASS_NAMES'][class_id]
        cv2.rectangle(resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(resized, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    plt.imshow(resized_rgb)
    plt.title("Detection Results")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    detect()
