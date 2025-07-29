### Project Overview: Mask Detection with YOLOv8

This project implements a complete pipeline for training a YOLOv8 object detection model to detect whether people are wearing masks. The solution includes data augmentation, model training, evaluation, and configuration management, providing an end-to-end workflow for creating a robust mask detection system.

#### Key Components:
1. **Data Augmentation Pipeline**  
   - Generates augmented training data using transformations:
     - Random horizontal flips (adjusts bounding boxes)
     - Color jitter (brightness/contrast/saturation)
   - Preserves YOLO label formatting during augmentation
   - Configurable augmentation factor (default: 3x expansion)

2. **YOLOv8 Training System**
   - Supports all model sizes (`n`, `s`, `m`, `l`, `x`)
   - Automated configuration with YAML generation
   - Advanced training parameters:
     - AdamW optimizer with LR=0.001
     - Mosaic (0.5) and MixUp (0.2) augmentation
     - Rotation (±10°) and scaling (0.5x)
     - Early stopping (patience=32 epochs)

3. **Evaluation Metrics**
   - Comprehensive validation reporting:
     - mAP@0.5 and mAP@0.5:0.95
     - Per-class precision/recall
     - Confusion matrix

#### Dataset Structure:
```bash
mask_dataset/
├── train/
│   ├── images/  # .jpg, .png
│   └── labels/  # .txt (YOLO format)
├── val/
│   ├── images/
│   └── labels/
└── test/        # Optional
```

#### Technical Specifications:
- **Classes**: `["mask", "no_mask"]`
- **Image Size**: 640x640 (configurable)
- **Augmentations**:
  ```python
  transform = Compose([
      RandomHorizontalFlip(p=0.5),
      ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
  ])
  ```
- **Training Defaults**:
  - Epochs: 128
  - Batch Size: 8 (adjust for GPU memory)
  - Model: YOLOv8l (large variant)

#### Dependencies:
```python
ultralytics==8.0.0      # YOLOv8 implementation
torch>=2.0.0            # Deep learning framework
torchvision>=0.15.0     # Image transformations
opencv-python>=4.7.0    # Image processing
numpy>=1.24.0           # Numerical operations
PyYAML>=6.0             # Configuration handling
Pillow>=9.0.0           # Image loading
```

#### Usage Workflow:
1. **Organize dataset** in specified directory structure
2. **Configure parameters** in `main()`:
   ```python
   DATA_DIR = Path("mask_dataset")
   CLASS_NAMES = ["mask", "no_mask"]
   ```
3. **Run pipeline**:
   ```bash
   python main.py
   ```

#### Key Features:
- **Reproducibility**: Seed control for all RNG operations
- **GPU Acceleration**: Automatic CUDA detection
- **Model Management**:
  - Saves best weights automatically
  - Periodic checkpoints (every 32 epochs)
- **Visual Feedback**: Progress tracking with metrics logging

#### Sample Output:
```
随机种子设置为: 42
==================================================
PyTorch版本: 2.1.0
CUDA可用: True
GPU设备: NVIDIA RTX 4090
==================================================
数据集配置文件已创建: mask_dataset/mask_dataset.yaml
开始训练YOLO模型 (尺寸: l, 轮数: 128)...
Epoch 1/128: 100%|████| 120/120 [01:23<00:00]
训练完成! 最佳模型保存于: runs/detect/mask_detection_yolov8l/weights/best.pt
评估模型性能...
评估结果:
  mAP@0.5: 0.9523
  mAP@0.5:0.95: 0.8721
  平均精确度 (Precision): 0.9342
  平均召回率 (Recall): 0.9125
  类别0精确度: 0.9567
  类别1精确度: 0.9118
```

This implementation provides a production-ready solution for mask detection that can be adapted to similar object detection tasks with minimal modifications. The modular design allows easy customization of augmentation strategies, model architectures, and training parameters.
