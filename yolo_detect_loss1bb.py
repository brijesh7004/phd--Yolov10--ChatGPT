import torch
import cv2
import numpy as np
from torchvision.ops import nms
from sklearn.cluster import AgglomerativeClustering
from yolov10_chatGPT2b import YOLOv10

# Your class names
CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']  # Edit this!

# Load model
def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = YOLOv10(variant='m', num_classes=5)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

# Preprocess image
def preprocess_image(image, input_size=640):
    img = cv2.resize(image, (input_size, input_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)

# Postprocess with confidence threshold & NMS
def postprocess(pred, conf_thresh=0.3, iou_thresh=0.5):
    # Flatten to [N, features]
    if pred.dim() == 4:
        pred = pred.view(-1, pred.shape[-1])
    elif pred.dim() == 3:
        pred = pred.squeeze(0)  # From [1, N, F] to [N, F]

    boxes = pred[:, :4]
    scores = pred[:, 4]
    class_probs = pred[:, 5:]
    class_ids = torch.argmax(class_probs, dim=1)
    confidences = scores * class_probs[range(len(class_ids)), class_ids]

    mask = confidences > conf_thresh
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Apply NMS
    keep = nms(boxes, confidences, iou_thresh)
    return boxes[keep], confidences[keep], class_ids[keep]


# Merge clusters per class
def cluster_and_merge_boxes(boxes, scores, class_ids, distance_thresh=20):
    merged_boxes = []
    merged_scores = []
    merged_class_ids = []

    for cls in class_ids.unique():
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        if cls_boxes.size(0) == 0:
            continue

        centers = ((cls_boxes[:, 0] + cls_boxes[:, 2]) / 2,
                   (cls_boxes[:, 1] + cls_boxes[:, 3]) / 2)
        centers_np = torch.stack(centers, dim=1).cpu().numpy()

        if len(centers_np) <= 1:
            merged_boxes.append(cls_boxes[0])
            merged_scores.append(cls_scores[0])
            merged_class_ids.append(cls.item())
            continue

        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_thresh).fit(centers_np)
        labels = clustering.labels_

        for lbl in np.unique(labels):
            cluster_boxes = cls_boxes[labels == lbl]
            cluster_scores = cls_scores[labels == lbl]

            x1 = cluster_boxes[:, 0].min()
            y1 = cluster_boxes[:, 1].min()
            x2 = cluster_boxes[:, 2].max()
            y2 = cluster_boxes[:, 3].max()

            merged_boxes.append(torch.tensor([x1, y1, x2, y2]))
            merged_scores.append(cluster_scores.mean())
            merged_class_ids.append(cls.item())

    if len(merged_boxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.stack(merged_boxes), torch.tensor(merged_scores), torch.tensor(merged_class_ids)

# Draw final boxes
def draw_boxes(image, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        if cls_id>5:
            continue
        label = f'{CLASS_NAMES[cls_id]} {score:.2f}'        
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image


# Load your model
model = load_model('best_yolov10_pc.pth')

# Load image
image = cv2.imread('assets/cars.jpg')
input_tensor = preprocess_image(image)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]  # Use only the primary prediction output

# Postprocess
boxes, scores, class_ids = postprocess(output)

# Merge overlapping boxes of same class
boxes, scores, class_ids = cluster_and_merge_boxes(boxes, scores, class_ids, distance_thresh=30)

# Draw
output_image = draw_boxes(image.copy(), boxes, scores, class_ids)

# Show result
cv2.imwrite("output_result.jpg", output_image)
print("Output image saved as 'output_result.jpg'")
