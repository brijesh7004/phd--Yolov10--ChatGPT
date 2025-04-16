"""
YOLOv10 Configuration
"""

import torch
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data3'
# WEIGHTS_DIR = BASE_DIR / 'weights'
# RESULTS_DIR = BASE_DIR / 'results'

# # Ensure directories exist
# WEIGHTS_DIR.mkdir(exist_ok=True)
# RESULTS_DIR.mkdir(exist_ok=True)

# # Default model weights
# MODEL_WEIGHTS = str(WEIGHTS_DIR / "yolov10b.pt")

# Device configuration
DEVICE_CONFIG = {
    "CUDA": True,      # Enable CUDA if available
    "AMP": True,       # Enable Automatic Mixed Precision
}

# Model configuration
MODEL_CONFIG = {
    # COCO class names
    # "CLASS_NAMES": [ "car", "motorcycle", "person" ],
    # "CLASS_NAMES": [ "Class 1", "Class 2", "Class 3", "Class 4", "Class 5" ],
    "CLASS_NAMES": [ "Person", "Motorcycle", "Car", "Truck", "Bus"],
    "NUM_CLASSES": 5,
    
    # Anchor configuration (3 anchors per scale, in [w, h] format)
    "ANCHORS": [
        [[10, 13], [16, 30], [33, 23]],       # Small objects
        [[30, 61], [62, 45], [59, 119]],      # Medium objects
        [[116, 90], [156, 198], [373, 326]]   # Large objects
    ],
    
    # Detection parameters
    "STRIDES": [8, 16, 32],
    "ANCHOR_PER_SCALE": 3,
    "IMG_SIZE": 640,
    "CONF_THRESHOLD": 0.25,
    "IOU_THRESHOLD": 0.45,
    "NUM_WORKERS": 4,
}

# Training configuration
TRAIN_CONFIG = {
    'MODEL_VARIANT': 'm',
    'BATCH_SIZE': 4,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 0.0005,
    'MOMENTUM': 0.937,
    'WARMUP_EPOCHS': 3,
    'SAVE_INTERVAL': 10,
    'EVAL_INTERVAL': 5,
    'PRETRAINED_WEIGHTS': None,  # Path to pretrained weights if any
    # 'SAVE_DIR': str(WEIGHTS_DIR),
    'RESUME': False,  # Whether to resume from last checkpoint
}

# Data configuration
DATA_CONFIG = {
    'TRAIN_IMG_DIR': str(DATA_DIR / 'images' / 'train'),
    'VAL_IMG_DIR': str(DATA_DIR / 'images' / 'val'),
    'TEST_IMG_DIR': str(DATA_DIR / 'images' / 'test'),
    'TRAIN_LABEL_DIR': str(DATA_DIR / 'labels' / 'train'),
    'VAL_LABEL_DIR': str(DATA_DIR / 'labels' / 'val'),
    'TEST_LABEL_DIR': str(DATA_DIR / 'labels' / 'test'),
}
