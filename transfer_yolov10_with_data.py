import torch
import torch.nn as nn
import torch.optim as optim
from yolov10_chatGPT2b import YOLOv10  # Assuming your model is defined in this file
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm  # Importing tqdm for progress bar

from credentials import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG

# Hyperparameters
batch_size = 16
learning_rate = 1e-4
epochs = 10
MODEL_WEIGHTS = 'transfer_learning/yolov10n.pt'


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


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize your YOLOv10 model (adjust based on your model's structure)
model = YOLOv10(variant=TRAIN_CONFIG['MODEL_VARIANT'], num_classes=MODEL_CONFIG['NUM_CLASSES'])  # Assuming 5 classes for detection

# Load pretrained weights with shape check
def load_pretrained_weights(model, weights_path):
    print("\nLoading pretrained weights from:", weights_path)
    
    pretrained = torch.load(weights_path, map_location=device)

    # Extract the state_dict properly
    if isinstance(pretrained, dict):
        if 'model' in pretrained:
            pretrained_dict = pretrained['model'].state_dict() if hasattr(pretrained['model'], 'state_dict') else pretrained['model']
        elif 'state_dict' in pretrained:
            pretrained_dict = pretrained['state_dict']
        else:
            pretrained_dict = pretrained  # Might already be a state_dict
    else:
        pretrained_dict = pretrained.state_dict() if hasattr(pretrained, 'state_dict') else pretrained

    model_dict = model.state_dict()
    matched_dict = {}

    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched_dict[k] = v
        else:
            print(f"Skipping unmatched key: {k} => pretrained: {v.shape}, model: {model_dict.get(k, 'None')}")

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    print("✅ Pretrained weights loaded with partial match.")



# Load the pretrained weights
load_pretrained_weights(model, MODEL_WEIGHTS)

# Move the model to the selected device (GPU or CPU)
model = model.to(device)

# Define your custom dataset (make sure to implement it based on your data format)
# train_dataset = CustomDataset('train_data_path', transform=None)  # Modify as needed
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataset = YoloDataset(DATA_CONFIG['TRAIN_IMG_DIR'], DATA_CONFIG['TRAIN_LABEL_DIR'])
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=True, num_workers=MODEL_CONFIG['NUM_WORKERS'], collate_fn=collate_fn)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Modify based on your task (e.g., object detection loss)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_dataloader:
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss (make sure to adjust based on your specific task)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_dataloader):.4f}')

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_model.pth')
print("Model fine-tuned and saved.")
