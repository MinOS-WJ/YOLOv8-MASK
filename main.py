import os
import shutil
import cv2
import numpy as np
import torch
import torchvision
import yaml
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# 设置随机种子确保可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")

# 数据增强和预处理
def augment_images_and_labels(data_dir, output_dir, augment_factor=2):
    """对训练数据进行增强"""
    train_img_dir = Path(data_dir) / "images"
    train_label_dir = Path(data_dir) / "labels"
    
    # 创建输出目录
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # 检查原始数据是否存在
    if not train_img_dir.exists() or not train_label_dir.exists():
        print(f"错误: 原始数据目录不存在")
        print(f"图像目录: {train_img_dir} {'存在' if train_img_dir.exists() else '不存在'}")
        print(f"标签目录: {train_label_dir} {'存在' if train_label_dir.exists() else '不存在'}")
        return
    
    # 获取原始图像文件
    image_files = list(train_img_dir.glob("*.*"))
    if not image_files:
        print(f"警告: 在 {train_img_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个原始图像")
    
    # 复制原始文件
    for img_file in image_files:
        shutil.copy(img_file, output_dir / "images")
    
    for label_file in train_label_dir.glob("*.txt"):
        shutil.copy(label_file, output_dir / "labels")
    
    print(f"开始数据增强 (因子: {augment_factor})...")
    
    # 增强变换
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # 应用增强
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            base_name = img_file.stem
            
            # 读取对应的标签
            label_file = train_label_dir / f"{base_name}.txt"
            if not label_file.exists():
                print(f"警告: 标签文件 {label_file} 不存在, 跳过增强")
                continue
                
            for i in range(augment_factor):
                # 应用变换
                augmented_img = transform(img)
                
                # 保存增强后的图像
                new_img_name = f"{base_name}_aug_{i}{img_file.suffix}"
                augmented_img.save(output_dir / "images" / new_img_name)
                
                # 复制标签文件（水平翻转需要调整坐标）
                if i == 0:  # 只复制标签（水平翻转需要特殊处理）
                    shutil.copy(label_file, output_dir / "labels" / f"{new_img_name}.txt")
                else:
                    # 对于水平翻转，调整边界框坐标
                    if np.random.rand() > 0.5:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                cls = int(parts[0])
                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                # 水平翻转：x坐标变为 1 - x
                                x = 1.0 - x
                                new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                            except ValueError:
                                continue
                        
                        with open(output_dir / "labels" / f"{new_img_name}.txt", 'w') as f:
                            f.writelines(new_lines)
                    else:
                        shutil.copy(label_file, output_dir / "labels" / f"{new_img_name}.txt")
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {str(e)}")
    
    total_images = len(list((output_dir / "images").glob("*.*")))
    print(f"数据增强完成! 总图像数: {total_images}")

# 创建YOLO数据集配置文件
def create_yaml_config(data_dir, class_names):
    """创建数据集YAML配置文件"""
    data_dir = Path(data_dir).resolve()
    
    # 检查数据集目录是否存在
    if not data_dir.exists():
        print(f"错误: 数据集目录 {data_dir} 不存在")
        return None
    
    config = {
        'path': str(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # 检查测试集是否存在
    test_dir = data_dir / "test" / "images"
    if test_dir.exists():
        config['test'] = 'test/images'
    
    yaml_path = data_dir / "mask_dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"数据集配置文件已创建: {yaml_path}")
    print(f"配置内容: {config}")
    return yaml_path

# 训练YOLO模型
def train_yolo_model(data_yaml, model_size="s", epochs=128, imgsz=640, batch=16):
    """训练YOLO模型"""
    print(f"\n开始训练YOLO模型 (尺寸: {model_size}, 轮数: {epochs})...")
    
    # 检查配置文件是否存在
    if not Path(data_yaml).exists():
        print(f"错误: 配置文件 {data_yaml} 不存在")
        return None
    
    # 加载模型
    try:
        model = YOLO(f"yolov8{model_size}.pt")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None
    
    # 训练参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 训练模型
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            name=f"mask_detection_yolov8{model_size}",
            patience=32,  # 早停轮数
            save=True,
            save_period=32,  # 每32个epoch保存一次
            optimizer="AdamW",
            lr0=0.001,
            augment=True,
            mosaic=0.5,  # 使用mosaic数据增强
            mixup=0.2,    # 使用mixup数据增强
            degrees=10.0, # 旋转角度范围
            scale=0.5     # 图像缩放范围
        )
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        return None
    
    # 返回最佳模型路径
    best_model = Path("runs/detect") / f"mask_detection_yolov8{model_size}" / "weights" / "best.pt"
    if best_model.exists():
        print(f"\n训练完成! 最佳模型保存于: {best_model}")
        return best_model
    else:
        print("\n训练完成，但未找到最佳模型文件")
        return None

# 修改后的评估函数
def evaluate_model(model_path, data_dir):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    if model_path is None or not Path(model_path).exists():
        print("错误: 模型文件不存在")
        return
    
    # 加载模型
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    # 在验证集上评估
    val_img_dir = Path(data_dir) / "val" / "images"
    if not val_img_dir.exists():
        print("验证集不存在，跳过评估")
        return
    
    try:
        metrics = model.val(
            data=str(Path(data_dir) / "mask_dataset.yaml"),
            split="val",
            imgsz=640,
            batch=16,
            conf=0.5,
            iou=0.6
        )
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        return
    
    # 打印评估结果 - 修正属性访问
    print("\n评估结果:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")          # 使用 map50
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")       # 使用 map
    print(f"  平均精确度 (Precision): {metrics.box.mp:.4f}")
    print(f"  平均召回率 (Recall): {metrics.box.mr:.4f}")
    
    # 打印每个类别的精确度
    for i, p_val in enumerate(metrics.box.p):
        print(f"  类别{i}精确度: {p_val:.4f}")
    
    return metrics

# 主函数
def main():
    # 设置参数
    DATA_DIR = Path("mask_dataset")  # 数据集目录
    CLASS_NAMES = ["mask", "no_mask"]  # 类别名称
    AUGMENTED_DIR = DATA_DIR / "train_aug"  # 增强后数据目录
    
    # 1. 初始化
    set_seed(42)
    print("="*50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print("="*50)
    
    # 2. 数据增强 - 可选步骤
    # augment_images_and_labels(DATA_DIR / "train", AUGMENTED_DIR, augment_factor=3)
    
    # 3. 创建数据集配置
    data_yaml = create_yaml_config(DATA_DIR, CLASS_NAMES)
    if data_yaml is None:
        print("无法创建YAML配置，程序退出")
        return
    
    # 4. 训练模型
    model_path = train_yolo_model(
        data_yaml,
        model_size="l",  # 模型尺寸: n, s, m, l, x
        epochs=128,      # 训练轮数
        imgsz=640,       # 图像大小
        batch=8         # 批大小
    )
    
    # 5. 评估模型
    if model_path:
        evaluate_model(model_path, DATA_DIR)
    
    print("\n项目完成! 所有任务均在命令行中执行。")

if __name__ == "__main__":
    main()
    
'''
mask_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/ (可选)
│   ├── images/
│   └── labels/
└── mask_dataset.yaml (自动生成)
'''