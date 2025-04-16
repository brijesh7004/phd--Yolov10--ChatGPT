import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolov10_chatGPT2b import YOLOv10
import torchvision.ops as ops
import os
import random
from credentials import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG

MODEL_WEIGHTS = 'best_yolov10_pc.pth'
CONF_THRESHOLD = 0.25  # Lowered back to 0.25 to capture more objects
NMS_THRESHOLD = 0.45
CLASS_AGNOSTIC_NMS = True  # Set to True to perform NMS across all classes
IMG_SIZE = 640
MAX_DETECTIONS = 100  # Reduced from 300 to show fewer, higher quality boxes
ENABLE_EXTRA_FILTERING = False  # Disable extra filtering that might be removing detections

def get_test_image(test_img_path=None):
    """
    Get a test image either from specified path or randomly from test directory
    """
    if test_img_path and os.path.exists(test_img_path):
        return test_img_path
    
    test_dir = DATA_CONFIG['TEST_IMG_DIR']
    if not os.path.exists(test_dir):
        test_dir = DATA_CONFIG['VAL_IMG_DIR']
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise FileNotFoundError("No image files found in test directory")

    random_image = random.choice(image_files)
    return os.path.join(test_dir, random_image)

def preprocess_image(img_path):
    """
    Load and preprocess image for YOLO model
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Store original image dimensions for later
    orig_h, orig_w = img.shape[:2]
    
    # Resize
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert BGR to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    return img, img_resized, img_tensor, (orig_h, orig_w)

def decode_outputs(outputs, num_classes):
    """
    Decode YOLO outputs to bounding box predictions
    """
    out_s, out_m, out_l = outputs
    
    # Decode each feature map
    decoded_s = decode_single_output(out_s, stride=8, num_classes=num_classes)
    decoded_m = decode_single_output(out_m, stride=16, num_classes=num_classes)
    decoded_l = decode_single_output(out_l, stride=32, num_classes=num_classes)
    
    # Concatenate all predictions
    predictions = torch.cat([decoded_s, decoded_m, decoded_l], dim=1)
    return predictions

def decode_single_output(output, stride, num_classes):
    """
    Decode a single feature map from YOLO output
    """
    B, C, H, W = output.shape
    
    # Reshape output
    output = output.view(B, 3, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()

    # Extract components
    cx = torch.sigmoid(output[..., 0])
    cy = torch.sigmoid(output[..., 1])
    w = output[..., 2]
    h = output[..., 3]
    obj_conf = torch.sigmoid(output[..., 4])
    class_probs = torch.sigmoid(output[..., 5:])

    # Create grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_x = grid_x.to(output.device).float()
    grid_y = grid_y.to(output.device).float()

    # Add offsets
    cx = (cx + grid_x.unsqueeze(0).unsqueeze(0)) * stride
    cy = (cy + grid_y.unsqueeze(0).unsqueeze(0)) * stride
    w = torch.exp(w) * stride
    h = torch.exp(h) * stride

    # Combine
    boxes = torch.stack([cx, cy, w, h, obj_conf], dim=-1)
    predictions = torch.cat([boxes, class_probs], dim=-1)
    
    # Reshape to [batch, num_boxes, num_values]
    return predictions.view(B, -1, 5 + num_classes)

def xywh_to_xyxy(boxes):
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def filter_overlapping_boxes(boxes, scores, classes, iou_threshold=0.65):
    """
    Filter boxes that are significantly overlapping with higher confidence boxes
    """
    if len(boxes) <= 1:
        return boxes, scores, classes
        
    n = len(boxes)
    filtered_indices = []
    
    # Create IoU matrix
    ious = ops.box_iou(boxes, boxes)
    
    # Get descending indices by score
    score_order = torch.argsort(scores, descending=True)
    
    # Mask of boxes to keep
    keep_mask = torch.ones(n, dtype=torch.bool)
    
    # Filter boxes
    for i in range(n):
        idx = score_order[i]
        
        # If this box is already filtered out, skip
        if not keep_mask[idx]:
            continue
            
        filtered_indices.append(idx)
        
        # Get overlap with remaining boxes
        overlap = ious[idx] > iou_threshold
        
        # For boxes with high overlap, check if they're the same class
        for j in range(n):
            if overlap[j] and keep_mask[j] and (classes[idx] == classes[j] or CLASS_AGNOSTIC_NMS):
                keep_mask[j] = False
                
        # Always keep the current box
        keep_mask[idx] = True
    
    return boxes[filtered_indices], scores[filtered_indices], classes[filtered_indices]

def non_max_suppression(predictions, conf_threshold=CONF_THRESHOLD, iou_threshold=NMS_THRESHOLD, max_detections=MAX_DETECTIONS):
    """
    Perform Non-Maximum Suppression on YOLO predictions
    """
    batch_size = predictions.shape[0]
    all_results = []
    
    for batch_idx in range(batch_size):
        print("\n=== Processing batch ===")
        batch_pred = predictions[batch_idx]
        print(f"Initial predictions: {batch_pred.shape[0]}")
        
        # Filter by objectness score first (much faster than filtering by class confidences)
        obj_conf = batch_pred[:, 4]
        obj_threshold = conf_threshold * 0.5  # Lower threshold for objectness
        mask = obj_conf > obj_threshold
        obj_count = mask.sum().item()
        print(f"After objectness filter (>{obj_threshold:.2f}): {obj_count}")
        
        if not mask.any():
            print("No detections passed objectness filter!")
            all_results.append([])
            continue
            
        batch_pred = batch_pred[mask]
        
        # Get class scores and ids
        class_scores, class_ids = batch_pred[:, 5:].max(1)
        
        # Calculate final confidence
        final_scores = batch_pred[:, 4] * class_scores
        
        # Apply confidence threshold to final scores
        conf_mask = final_scores > conf_threshold
        conf_count = conf_mask.sum().item()
        print(f"After confidence filter (>{conf_threshold:.2f}): {conf_count}")
        
        if not conf_mask.any():
            print("No detections passed confidence filter!")
            all_results.append([])
            continue
            
        final_scores = final_scores[conf_mask]
        class_ids = class_ids[conf_mask]
        boxes = xywh_to_xyxy(batch_pred[conf_mask, :4])
        
        # Apply class-aware NMS or class-agnostic NMS
        if CLASS_AGNOSTIC_NMS:
            # Class-agnostic NMS (consider all boxes regardless of class)
            keep_idx = ops.nms(boxes, final_scores, iou_threshold)
            nms_count = len(keep_idx)
            print(f"After NMS (IoU>{iou_threshold:.2f}): {nms_count}")
            
            # Take top scoring boxes
            if len(keep_idx) > max_detections:
                keep_idx = keep_idx[:max_detections]
                
            boxes = boxes[keep_idx]
            final_scores = final_scores[keep_idx]
            class_ids = class_ids[keep_idx]
            
            # Additional filtering to remove overlapping boxes
            if ENABLE_EXTRA_FILTERING:
                pre_filter_count = len(boxes)
                boxes, final_scores, class_ids = filter_overlapping_boxes(
                    boxes, final_scores, class_ids, iou_threshold=0.65
                )
                post_filter_count = len(boxes)
                print(f"After extra filtering: {post_filter_count} (removed {pre_filter_count - post_filter_count})")
        else:
            # Class-aware NMS (process each class separately)
            keep_boxes = []
            keep_scores = []
            keep_classes = []
            
            class_ids_unique = class_ids.unique()
            print(f"Unique classes found: {len(class_ids_unique)}")
            
            for cls_id in class_ids_unique:
                cls_mask = class_ids == cls_id
                cls_count = cls_mask.sum().item()
                cls_name = MODEL_CONFIG['CLASS_NAMES'][cls_id.item()] if cls_id.item() < len(MODEL_CONFIG['CLASS_NAMES']) else f"Unknown-{cls_id.item()}"
                print(f"  Class {cls_name}: {cls_count} candidates")
                
                if not cls_mask.any():
                    continue
                    
                cls_boxes = boxes[cls_mask]
                cls_scores = final_scores[cls_mask]
                
                # Apply NMS
                keep_idx = ops.nms(cls_boxes, cls_scores, iou_threshold)
                keep_count = len(keep_idx)
                print(f"    After NMS: {keep_count}")
                
                # Take top scoring boxes per class
                if len(keep_idx) > max_detections // 2:
                    keep_idx = keep_idx[:max_detections // 2]
                
                keep_boxes.append(cls_boxes[keep_idx])
                keep_scores.append(cls_scores[keep_idx])
                keep_classes.append(torch.full_like(keep_idx, cls_id))
            
            # Combine results
            if keep_boxes:
                boxes = torch.cat(keep_boxes)
                final_scores = torch.cat(keep_scores)
                class_ids = torch.cat(keep_classes)
                print(f"Total kept after class-aware NMS: {len(boxes)}")
            else:
                print("No boxes kept after class-aware NMS!")
                all_results.append([])
                continue
        
        # Limit total number of detections
        if len(boxes) > max_detections:
            top_k = torch.argsort(final_scores, descending=True)[:max_detections]
            boxes = boxes[top_k]
            final_scores = final_scores[top_k]
            class_ids = class_ids[top_k]
            print(f"Limited to top {max_detections} detections")
        
        final_count = len(boxes)
        print(f"Final detection count: {final_count}")
        
        if final_count == 0:
            print("WARNING: No detections remained after all filtering!")
            all_results.append([])
            continue
            
        # Combine boxes, scores, and class ids
        batch_result = torch.cat([
            boxes, 
            final_scores.unsqueeze(1), 
            class_ids.float().unsqueeze(1)
        ], dim=1)
        
        all_results.append(batch_result)
    
    return all_results

def draw_boxes(image, detections, class_names):
    """
    Draw bounding boxes on the image
    """
    img = image.copy()
    
    if len(detections) == 0:
        return img
    
    # Use a fixed color palette for better visualization
    color_palette = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (128, 255, 0),  # Light green
        (255, 128, 0),  # Light blue
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Orange
    ]
        
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        
        # Use class id to determine color (cycle through palette)
        color = color_palette[cls_id % len(color_palette)]
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown-{cls_id}"
        label = f'{class_name} {conf:.2f}'
        
        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(
            img, 
            (x1, y1 - label_height - baseline), 
            (x1 + label_width, y1),
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            img, 
            label, 
            (x1, y1 - baseline), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
    
    return img

def detect(img_path=None, conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD):
    """
    Run object detection on an image
    """
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    model = YOLOv10(
        variant=TRAIN_CONFIG['MODEL_VARIANT'], 
        num_classes=MODEL_CONFIG['NUM_CLASSES']
    ).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try direct loading in case it's just the state dict
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {MODEL_WEIGHTS}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Set model to evaluation mode
    model.eval()

    # Get and preprocess image
    img_path = get_test_image(img_path)
    print(f"Processing image: {img_path}")
    try:
        original_img, resized_img, img_tensor, orig_dims = preprocess_image(img_path)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return
    
    # Run inference
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        try:
            outputs = model(img_tensor)
            print("Model inference successful")
        except Exception as e:
            print(f"Error during model inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    # Decode outputs
    try:
        predictions = decode_outputs(outputs, MODEL_CONFIG['NUM_CLASSES'])
        print(f"Decoded predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Error decoding outputs: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Print prediction shape and confidence distribution
    confidences = predictions[0, :, 4]
    high_conf = (confidences > 0.8).sum().item()
    med_conf = ((confidences > 0.5) & (confidences <= 0.8)).sum().item()
    low_conf = ((confidences > 0.3) & (confidences <= 0.5)).sum().item()
    very_low_conf = ((confidences > 0.1) & (confidences <= 0.3)).sum().item()
    
    print(f"Confidence distribution:")
    print(f"  > 0.8: {high_conf}")
    print(f"  0.5-0.8: {med_conf}")
    print(f"  0.3-0.5: {low_conf}")
    print(f"  0.1-0.3: {very_low_conf}")
    
    # Run NMS
    try:
        detections = non_max_suppression(
            predictions, 
            conf_threshold=conf_threshold,
            iou_threshold=nms_threshold,
            max_detections=MAX_DETECTIONS
        )[0]  # Batch size = 1
    except Exception as e:
        print(f"Error during NMS: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Print detection summary
    if isinstance(detections, list) or len(detections) == 0:
        print("No objects detected in final output!")
        
        # Draw a basic grid on the image to show it's being processed
        result_img = resized_img.copy()
        h, w = result_img.shape[:2]
        for i in range(0, w, 50):
            cv2.line(result_img, (i, 0), (i, h), (200, 200, 200), 1)
        for i in range(0, h, 50):
            cv2.line(result_img, (0, i), (w, i), (200, 200, 200), 1)
            
        # Add text explaining no detections
        cv2.putText(
            result_img,
            "No detections with current settings",
            (w//2 - 150, h//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
    else:
        print(f"Detected {len(detections)} objects")
        
        # Print class distribution
        if len(detections) > 0:
            class_counts = {}
            for det in detections:
                cls_id = int(det[5])
                class_name = MODEL_CONFIG['CLASS_NAMES'][cls_id] if cls_id < len(MODEL_CONFIG['CLASS_NAMES']) else f"Unknown-{cls_id}"
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
                    
            print("Class distribution:")
            for cls_name, count in class_counts.items():
                print(f"  {cls_name}: {count}")
        
        # Draw boxes on image
        result_img = draw_boxes(
            resized_img, 
            detections, 
            MODEL_CONFIG['CLASS_NAMES']
        )
    
    # Display results
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12))
    plt.imshow(result_rgb)
    if isinstance(detections, list) or len(detections) == 0:
        plt.title("No objects detected")
    else:
        plt.title(f"Detection Results: {len(detections)} objects")
    plt.axis('off')
    plt.tight_layout()
    
    # Save output image
    output_path = 'yolo_detection_result.jpg'
    cv2.imwrite(output_path, result_img)
    print(f"Output image saved as '{output_path}'")
    
    # Try a lower threshold version if no detections
    if (isinstance(detections, list) or len(detections) == 0) and conf_threshold > 0.1:
        print("\n\nTrying with a lower confidence threshold...")
        lower_threshold = max(0.1, conf_threshold - 0.1)
        alt_output_path = f'yolo_detection_result_conf{int(lower_threshold*100)}.jpg'
        try:
            alt_detections = non_max_suppression(
                predictions, 
                conf_threshold=lower_threshold,
                iou_threshold=nms_threshold,
                max_detections=MAX_DETECTIONS
            )[0]
            
            if not isinstance(alt_detections, list) and len(alt_detections) > 0:
                print(f"Found {len(alt_detections)} objects with lower threshold {lower_threshold}")
                alt_result_img = draw_boxes(resized_img, alt_detections, MODEL_CONFIG['CLASS_NAMES'])
                cv2.imwrite(alt_output_path, alt_result_img)
                print(f"Alternative detection saved as '{alt_output_path}'")
        except Exception as e:
            print(f"Error creating alternative detection: {str(e)}")
    
    # Show plot
    plt.savefig('detection_plot.png')
    plt.show()

    return detections, result_img

if __name__ == "__main__":
    detect()
