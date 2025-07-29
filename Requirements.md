**requirements.txt:**
```
ultralytics==8.2.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.4
numpy>=1.20.0
PyYAML>=6.0
Pillow>=9.0.0
```

**Key dependencies explained:**
1. `ultralytics`: Required for YOLOv8 model implementation
2. `torch` and `torchvision`: Core deep learning framework
3. `opencv-python`: Image processing and computer vision operations
4. `numpy`: Numerical operations and array handling
5. `PyYAML`: For dataset configuration file handling
6. `Pillow`: Image loading and basic transformations

**Additional notes:**
- The script requires CUDA-enabled GPU for optimal training performance
- Dataset should be organized in YOLO format with separate train/val/test folders
- Input images should be in common formats (JPG, PNG, etc.)
- Label files should be in YOLO .txt format with normalized coordinates

**Dataset directory structure:**
```
mask_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/ (optional)
│   ├── images/
│   └── labels/
└── mask_dataset.yaml (auto-generated)
```

**Hardware recommendations:**
- NVIDIA GPU (8GB+ VRAM recommended for YOLOv8l)
- 16GB+ RAM
- Sufficient storage space for augmented datasets
