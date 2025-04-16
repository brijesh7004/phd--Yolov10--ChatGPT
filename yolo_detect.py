import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolov10_chatGPT2b import YOLOv10
import torchvision.ops as ops  # For NMS
import os
import random

from credentials import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG
MODEL_WEIGHTS = 'best_yolov10.pth'


def get_random_test_image():
    test_dir = DATA_CONFIG['TEST_IMG_DIR']
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise FileNotFoundError("No image files found in test directory")

    random_image = random.choice(image_files)
    return os.path.join(test_dir, random_image)


# ---------- Non-Max Suppression ----------
def non_max_suppression(predictions, conf_threshold=0.5, nms_threshold=0.3):
    boxes, confidences, class_ids = [], [], []

    for pred in predictions[0]:  # batch size = 1
        class_scores = pred[5:]
        class_id = torch.argmax(class_scores).item()
        class_prob = class_scores[class_id]
        conf = pred[4]
        score = conf * class_prob

        # print(f"conf: {conf:.2f}, class_prob: {class_prob:.4f}, score: {score:.4f}")
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
    confidences = torch.tensor(confidences)
    class_ids = torch.tensor(class_ids)

    keep = ops.nms(boxes, confidences, nms_threshold)
    return boxes[keep], confidences[keep], class_ids[keep]


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
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
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

    boxes, confidences, class_ids = non_max_suppression(predictions, conf_threshold=threshold)
    print("NMS Results:")
    print("Boxes:", boxes)
    print("Confidences:", confidences)
    print("Class IDs:", class_ids)

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = MODEL_CONFIG['CLASS_NAMES'][class_id]
        cv2.rectangle(resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(resized, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    plt.imshow(resized_rgb)
    plt.title("Detection Results")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    detect()
