import torch
from yolov10_chatGPT2b import YOLOv10
from torchsummary import summary

model = YOLOv10(variant='n', num_classes=5)
summary(model, input_size=(3, 640, 640))
print(model)


# # Load pretrained checkpoint
# ckpt = torch.load("yolov10n.pt", map_location="cpu")

# # Extract model state_dict only
# if 'model' in ckpt:
#     pretrained_state = ckpt['model'].float().state_dict()
# else:
#     pretrained_state = ckpt

# # Initialize model with expected structure
# model = YOLOv10(variant='n', num_classes=80)
# model.load_state_dict(pretrained_state, strict=False)

# # Show model summary
# summary(model, input_size=(3, 640, 640))

# Load the pretrained YOLOv10n checkpoint
ckpt = torch.load("yolov10n.pt", map_location="cpu")

# Extract the embedded model
model = ckpt['model'].float() if 'model' in ckpt else ckpt

# Print full architecture
print(model)

# Optional: Print number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
