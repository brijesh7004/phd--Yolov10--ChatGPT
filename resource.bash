# Create a virtual environment (recommended)
python -m venv yolov10_env
source 'yolov10_env\Scripts\activate' #yolov10_env/bin/activate  # On Windows use `yolov10_env\Scripts\activate`

# Install essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python matplotlib tqdm pyyaml pandas seaborn
pip install ultralytics  # For some utilities we'll borrow