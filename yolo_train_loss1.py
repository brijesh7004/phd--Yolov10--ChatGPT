from datetime import datetime
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import glob
import numpy as np
# import matplotlib.pyplot as plt
from yolov10_chatGPT2b import YOLOv10
from tqdm import tqdm  # Importing tqdm for progress bar

from credentials import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
        self.label_paths = sorted(glob.glob(f"{label_dir}/*.txt"))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        img = torch.from_numpy(img.copy()).float() / 255.0

        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, cx, cy, w, h = map(float, line.strip().split())
                labels.append([cls, cx, cy, w, h])
        labels = torch.tensor(labels)

        return img, labels

###########################################################################################
################## Loss Function-1 #######################
#####################################################################################
# class YOLOLoss(nn.Module):
#     def __init__(self, num_classes=MODEL_CONFIG['NUM_CLASSES'], device='cuda'):
#         super(YOLOLoss, self).__init__()
#         self.num_classes = num_classes
#         self.num_anchors = 3
#         self.device = device
#         self.bce_cls = nn.BCEWithLogitsLoss()
#         self.bce_obj = nn.BCEWithLogitsLoss()
#         self.smooth_l1 = nn.SmoothL1Loss()

#     def forward(self, preds, targets):
#         """
#         preds: list of [B, anchors*C, H, W] outputs at different scales
#         targets: list of labels [cls, x, y, w, h] for each object
#         """
#         total_loss = 0
#         for i, pred in enumerate(preds):
#             B, C, H, W = pred.shape  # shape: [B, anchors*C, H, W]
#             pred = pred.view(B, self.num_anchors, -1, H, W).permute(0, 1, 3, 4, 2).contiguous()
#             # Now shape: [B, A, H, W, C]
            
#             # Dummy loss calculation loop (to be replaced with actual objectness/iou/cls losses)
#             for b in range(B):
#                 for t in targets[b]:
#                     # Ensure the label is valid
#                     if len(t) != 5:
#                         print(f"Skipping invalid target {t}")
#                         continue
                    
#                     cls, x, y, w, h = t  # Fixed: no slicing

#                     # Grid cell location
#                     gx, gy = int(x * W), int(y * H)

#                     # Sample logic (placeholder for loss calculations)
#                     obj_pred = pred[b, 0, gy, gx, 0]  # Example: objectness score
#                     pred_x = pred[b, 0, gy, gx, 1]
#                     pred_y = pred[b, 0, gy, gx, 2]

#                     # Add dummy sub-losses
#                     total_loss += (obj_pred - 1.0) ** 2
#                     total_loss += (pred_x - x) ** 2
#                     total_loss += (pred_y - y) ** 2
#                     # TODO: Add width, height, class loss, IOU loss, etc.
#         return total_loss

#####################################################################################
################## Loss Function-2 #######################
#####################################################################################
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes=MODEL_CONFIG['NUM_CLASSES'], img_size=640):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors  # List of 3 anchors for this scale (e.g., [(10,13), (16,30), (33,23)])
        self.num_anchors = len(anchors)  # Should be 3
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.ignore_threshold = 0.5
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss()
        self.img_size = img_size

    def forward(self, pred, targets=None):
        # pred shape: [B, num_anchors*(5+num_classes), H, W]
        B, C, H, W = pred.shape
        assert C == self.num_anchors * self.bbox_attrs, f"Expected {self.num_anchors * self.bbox_attrs} channels but got {C}"

        # Reshape and permute to [B, num_anchors, H, W, bbox_attrs]
        pred = pred.view(B, self.num_anchors, self.bbox_attrs, H, W).permute(0, 1, 3, 4, 2).contiguous()

        # Extract components
        x = torch.sigmoid(pred[..., 0])  # center x
        y = torch.sigmoid(pred[..., 1])  # center y
        w = pred[..., 2]  # width
        h = pred[..., 3]  # height
        conf = torch.sigmoid(pred[..., 4])  # object confidence
        cls_pred = pred[..., 5:]  # class prediction

        if targets is None:
            # Inference mode – decode predictions
            scaled_anchors = [(a_w / self.img_size, a_h / self.img_size) for a_w, a_h in self.anchors]
            return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1), conf.unsqueeze(-1), cls_pred), -1)

        # Training mode – compute loss
        # Create masks and targets
        obj_mask = torch.zeros_like(conf, dtype=torch.bool)
        noobj_mask = torch.ones_like(conf, dtype=torch.bool)

        tx = torch.zeros_like(x)
        ty = torch.zeros_like(y)
        tw = torch.zeros_like(w)
        th = torch.zeros_like(h)
        tconf = torch.zeros_like(conf)
        tcls = torch.zeros_like(cls_pred)

        # Scale anchors to current feature map size
        stride = self.img_size / H
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]

        for b in range(B):
            for t in range(targets[b].shape[0]):
                if targets[b][t].sum() == 0:
                    continue
                gx, gy, gw, gh = targets[b][t][1:] * H  # assuming square input/output
                gi, gj = int(gx), int(gy)

                box = torch.tensor([0, 0, gw, gh]).unsqueeze(0)
                anchor_shapes = torch.tensor([[0, 0, aw, ah] for aw, ah in scaled_anchors])
                anch_ious = bbox_iou(box, anchor_shapes)

                best_n = torch.argmax(anch_ious).item()

                if gi < W and gj < H:
                    obj_mask[b, best_n, gj, gi] = 1
                    noobj_mask[b, best_n, gj, gi] = 0

                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    tw[b, best_n, gj, gi] = torch.log(gw / scaled_anchors[best_n][0] + 1e-16)
                    th[b, best_n, gj, gi] = torch.log(gh / scaled_anchors[best_n][1] + 1e-16)

                    tconf[b, best_n, gj, gi] = 1
                    tcls[b, best_n, gj, gi, int(targets[b][t][0])] = 1

        # Loss components
        loss_x = self.bce_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.bce_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(conf[noobj_mask], tconf[noobj_mask])
        loss_cls = self.bce_loss(cls_pred[obj_mask], tcls[obj_mask])


        lambda_coord = 5
        lambda_noobj = 0.5

        total_loss = (
            lambda_coord * (loss_x + loss_y + loss_w + loss_h) +
            loss_conf_obj +
            lambda_noobj * loss_conf_noobj +
            loss_cls
        ) / B

        # total_loss = (loss_x + loss_y + loss_w + loss_h + loss_conf_obj + loss_conf_noobj + loss_cls) / B
        return total_loss
    
def bbox_iou(box1, box2, x1y1x2y2=False):
    # Convert width/height to x1/y1/x2/y2 if needed
    if not x1y1x2y2:
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    return iou


#####################################################################################
################## Loss Function-3 #######################
#####################################################################################
# class YOLOLoss(nn.Module):
#     def __init__(self, anchors):
#         super(YOLOLoss, self).__init__()
#         self.anchors = anchors
#         self.num_anchors = len(anchors)
#         self.num_classes = MODEL_CONFIG['NUM_CLASSES']
#         self.mse_loss = nn.MSELoss(reduction='mean')
#         self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

#     def forward(self, predictions, targets):
#         # Dummy example (replace with your full implementation)
#         loss_x = self.mse_loss(predictions[..., 0], targets[..., 0])
#         loss_y = self.mse_loss(predictions[..., 1], targets[..., 1])
#         loss_w = self.mse_loss(predictions[..., 2], targets[..., 2])
#         loss_h = self.mse_loss(predictions[..., 3], targets[..., 3])
#         loss_conf_obj = self.bce_loss(predictions[..., 4], targets[..., 4])
#         loss_conf_noobj = self.bce_loss(predictions[..., 4], targets[..., 4])  # update based on mask logic
#         loss_cls = self.bce_loss(predictions[..., 5:], targets[..., 5:])

#         total_loss = (loss_x + loss_y + loss_w + loss_h +
#                       loss_conf_obj + loss_conf_noobj + loss_cls)

#         return total_loss, {
#             'x': loss_x.item(),
#             'y': loss_y.item(),
#             'w': loss_w.item(),
#             'h': loss_h.item(),
#             'conf_obj': loss_conf_obj.item(),
#             'conf_noobj': loss_conf_noobj.item(),
#             'cls': loss_cls.item()
#         }



def collate_fn(batch):
    imgs, labels = zip(*batch)
    
    # Find the maximum number of boxes in the batch
    max_boxes = max([label.size(0) if label.dim() > 0 else 0 for label in labels])

    padded_labels = []
    for label in labels:
        if label.dim() == 1:  # If the label is 1D (for example, no boxes)
            # We assume 1D labels are empty or invalid, so we pad them
            padded_labels.append(torch.zeros((max_boxes, 5)))  # Assuming 5 values per label (x, y, w, h, class)
        elif label.dim() == 0:  # Handle empty labels (edge case)
            padded_labels.append(torch.zeros((max_boxes, 5)))  # No boxes, padding to max_boxes
        else:
            # If label is 2D (normal case)
            padding = torch.zeros((max_boxes - label.size(0), label.size(1)))  # Pad to max_boxes
            padded_label = torch.cat((label, padding), dim=0)  # Concatenate label with padding
            padded_labels.append(padded_label)

    # Stack padded labels into a single tensor
    padded_labels = torch.stack(padded_labels, dim=0)

    # Stack the images into a single tensor
    imgs = torch.stack(imgs, dim=0)

    return imgs, padded_labels



def draw_boxes(img, boxes):
    img = img.copy()
    for box in boxes:
        cls, cx, cy, w, h = box
        x1 = int((cx - w / 2) * img.shape[1])
        y1 = int((cy - h / 2) * img.shape[0])
        x2 = int((cx + w / 2) * img.shape[1])
        y2 = int((cy + h / 2) * img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box
    return img

def train(preTrainedModel=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv10(variant=TRAIN_CONFIG['MODEL_VARIANT'], num_classes=MODEL_CONFIG['NUM_CLASSES']).to(device)
    if preTrainedModel is not None:
        model.load_state_dict(torch.load(preTrainedModel))

    train_dataset = YoloDataset(DATA_CONFIG['TRAIN_IMG_DIR'], DATA_CONFIG['TRAIN_LABEL_DIR'])
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=True, num_workers=MODEL_CONFIG['NUM_WORKERS'], collate_fn=collate_fn)

    val_dataset = YoloDataset(DATA_CONFIG['VAL_IMG_DIR'], DATA_CONFIG['VAL_LABEL_DIR'])
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=True, num_workers=MODEL_CONFIG['NUM_WORKERS'], collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True) # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    anchors = [
        (17,22), (32,30), (36,66),     # small
        (61,50), (62,122), (116,92),   # medium
        (93,240), (213,160), (198,378) # large
    ]
    criterion = YOLOLoss(anchors=anchors)
    best_val_loss = float('inf')

    save_path = f"yolov10{TRAIN_CONFIG['MODEL_VARIANT']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    for epoch in range(TRAIN_CONFIG['EPOCHS']):
        model.train()
        total_loss = 0

        # Wrapping the data loader with tqdm to show progress
        progress_bar = tqdm(train_dataloader, desc=f"[Train] Epoch {epoch+1}/{TRAIN_CONFIG['EPOCHS']}", unit="batch", position=0, ncols=100)
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out_s, out_m, out_l = model(imgs)
             # Determine the target size (let's assume out_l has the largest spatial dimensions)
            target_size = out_l.shape[2:]  # (H, W) of the largest feature map

            # Upsample or downsample the feature maps to the target size
            out_s_resized = F.interpolate(out_s, size=target_size, mode='bilinear', align_corners=False)
            out_m_resized = F.interpolate(out_m, size=target_size, mode='bilinear', align_corners=False)
            out_l_resized = out_l

            output = torch.cat([out_s_resized, out_m_resized, out_l_resized], dim=1)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1), refresh=True)

            # # Show a sample image with bounding boxes every 100 batches
            # if progress_bar.n % 100 == 0:
            #     sample_img = imgs[0].cpu().numpy().transpose(1, 2, 0)
            #     sample_labels = labels[0].cpu().numpy()
            #     img_with_boxes = draw_boxes(sample_img, sample_labels)
            #     plt.imshow(img_with_boxes)
            #     plt.title(f"Epoch {epoch+1} - Batch {progress_bar.n}")
            #     plt.show()

        # ---------------- VALIDATION LOOP ----------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc=f"[Val] Validating...", unit="batch", position=1, ncols=100)
            for val_imgs, val_labels in val_bar:
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)

                out_s, out_m, out_l = model(val_imgs)
                out_s_resized = F.interpolate(out_s, size=target_size, mode='bilinear', align_corners=False)
                out_m_resized = F.interpolate(out_m, size=target_size, mode='bilinear', align_corners=False)
                out_l_resized = out_l
                val_output = torch.cat([out_s_resized, out_m_resized, out_l_resized], dim=1)

                loss_val = criterion(val_output, val_labels)
                val_loss += loss_val.item()
                val_bar.set_postfix(val_loss=val_loss / (val_bar.n + 1), refresh=True)
        
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"✅ Epoch {epoch+1} done: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}                ")

        # Validation loss computed
        scheduler.step(avg_val_loss)  # Pass validation loss to scheduler
        print(f"📉 Learning rate: {scheduler.optimizer.param_groups[0]['lr']}")

        # Save only if validation loss improves
        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            # torch.save(model, 'yolov10_model_complete.pth')   #save entire model with all weights
            print(f"✅ Best model updated at Epoch {epoch+1} with val loss {avg_val_loss:.4f} in {save_path}")

if __name__ == "__main__":
    train(preTrainedModel='yolov10m_20250415_1201.pth')
