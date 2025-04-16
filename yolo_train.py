from datetime import datetime
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import cv2
import glob
import numpy as np
# import matplotlib.pyplot as plt
from yolov10_chatGPT1b import YOLOv10
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

# def build_targets(preds, targets, anchors, num_classes):
#     """
#     Converts annotations into label tensors compatible with predictions.
#     """
#     batch_size, _, grid_size, _ = preds.shape
#     target_tensor = torch.zeros_like(preds)

#     for b in range(batch_size):
#         for t in range(targets[b].shape[0]):
#             gx, gy, gw, gh, cls = targets[b][t]
#             gi, gj = int(gx * grid_size), int(gy * grid_size)

#             for a in range(len(anchors)):
#                 # Assign values (you can refine IoU-based anchor assignment here)
#                 target_tensor[b, a * (5 + num_classes):(a + 1) * (5 + num_classes), gj, gi] = torch.tensor([
#                     gx, gy, gw, gh, 1.0, *torch.nn.functional.one_hot(cls.long(), num_classes).float()
#                 ])

#     return target_tensor

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=5, anchors=3, image_size=640):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = anchors
        self.image_size = image_size

        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0

    def build_targets(self, pred, targets):
        # pred: [B, H, W, A, 5+num_classes]
        B, H, W, A, _ = pred.shape
        device = pred.device

        target = torch.zeros((B, H, W, A, 5 + self.num_classes), device=device)

        for i in range(B):
            for obj in targets[i]:
                class_id, x, y, w, h = obj.tolist()
                grid_x = int(x * W)
                grid_y = int(y * H)

                anchor_idx = 0  # Simplified: always use first anchor
                if grid_x < W and grid_y < H:
                    target[i, grid_y, grid_x, anchor_idx, 0] = x
                    target[i, grid_y, grid_x, anchor_idx, 1] = y
                    target[i, grid_y, grid_x, anchor_idx, 2] = w
                    target[i, grid_y, grid_x, anchor_idx, 3] = h
                    target[i, grid_y, grid_x, anchor_idx, 4] = 1  # objectness
                    target[i, grid_y, grid_x, anchor_idx, 5 + int(class_id)] = 1  # class one-hot

        return target

    def forward(self, preds, targets):
        total_loss = 0
        loss_components = {}

        for i, pred in enumerate(preds):
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            pred = pred.view(B, H, W, self.num_anchors, 5 + self.num_classes)
            # shape = (B, H, W, self.num_anchors, 5 + self.num_classes)
            # pred = pred.view(*shape)

            target = self.build_targets(pred, targets)

            loss_x = self.lambda_coord * F.mse_loss(pred[..., 0], target[..., 0])
            loss_y = self.lambda_coord * F.mse_loss(pred[..., 1], target[..., 1])
            loss_w = self.lambda_coord * F.mse_loss(pred[..., 2], target[..., 2])
            loss_h = self.lambda_coord * F.mse_loss(pred[..., 3], target[..., 3])
            loss_obj = self.lambda_obj * F.binary_cross_entropy_with_logits(pred[..., 4], target[..., 4])

            loss_noobj = self.lambda_noobj * F.binary_cross_entropy_with_logits(pred[..., 4], target[..., 4], reduction='none')
            loss_noobj = (loss_noobj * (1 - target[..., 4])).mean()

            loss_class = self.lambda_obj * F.binary_cross_entropy_with_logits(pred[..., 5:], target[..., 5:], reduction='none')
            loss_class = (loss_class * target[..., 4:5]).mean()

            total_loss += loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_class

            loss_components[f'loss_x_{i}'] = loss_x
            loss_components[f'loss_y_{i}'] = loss_y
            loss_components[f'loss_w_{i}'] = loss_w
            loss_components[f'loss_h_{i}'] = loss_h
            loss_components[f'loss_obj_{i}'] = loss_obj
            loss_components[f'loss_noobj_{i}'] = loss_noobj
            loss_components[f'loss_class_{i}'] = loss_class

        return total_loss, loss_components


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

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv10(variant=TRAIN_CONFIG['MODEL_VARIANT'], num_classes=MODEL_CONFIG['NUM_CLASSES']).to(device)

    train_dataset = YoloDataset(DATA_CONFIG['TRAIN_IMG_DIR'], DATA_CONFIG['TRAIN_LABEL_DIR'])
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=True, num_workers=MODEL_CONFIG['NUM_WORKERS'], collate_fn=collate_fn)

    val_dataset = YoloDataset(DATA_CONFIG['VAL_IMG_DIR'], DATA_CONFIG['VAL_LABEL_DIR'])
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=True, num_workers=MODEL_CONFIG['NUM_WORKERS'], collate_fn=collate_fn)


    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    anchors = [
        (17,22), (32,30), (36,66),     # small
        (61,50), (62,122), (116,92),   # medium
        (93,240), (213,160), (198,378) # large
    ]
    criterion = YOLOLoss().to(device)
    best_val_loss = float('inf')

    save_path = f"yolov10{TRAIN_CONFIG['MODEL_VARIANT']}_{datetime.now().strftime('%Y%m%d')}.pth"
    for epoch in range(TRAIN_CONFIG['EPOCHS']):
        model.train()
        running_loss = 0
        loss_components = {}

        progress_bar = tqdm(train_dataloader, desc=f"[Train] Epoch {epoch+1}/{TRAIN_CONFIG['EPOCHS']}", unit="batch", position=0, ncols=100)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_components = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1), refresh=True)

            # if batch_idx % 10 == 0:
            #     print(f"Epoch [{epoch+1}/{TRAIN_CONFIG['EPOCHS']}], Batch [{batch_idx+1}/{len(dataloader)}], "
            #           f"Loss: {loss.item():.4f}, " + ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_components.items()]))
            
        # ---------------- VALIDATION LOOP ----------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc=f"[Val] Epoch {epoch+1}/{TRAIN_CONFIG['EPOCHS']}", unit="batch", position=1, ncols=100)
            for images, labels in val_bar:
                images = images.to(device)
                labels = [label.to(device) for label in labels]

                outputs = model(images)
                loss, _ = criterion(outputs, labels)
                val_loss += loss.item()
                val_bar.set_postfix(val_loss=val_loss / (val_bar.n + 1), refresh=True)
        
        avg_train_loss = running_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"✅ Epoch {epoch+1} done: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}                          ")

        # Validation loss computed
        scheduler.step()  # Pass validation loss to scheduler
        print(f"📉 Learning rate: {scheduler.optimizer.param_groups[0]['lr']}")

        # Save only if validation loss improves
        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model updated at Epoch {epoch+1} with val loss {avg_val_loss:.4f}")

if __name__ == "__main__":
    train()
