#训练模型

import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch.multiprocessing as mp
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights 

# 解决 Windows multiprocessing 问题
mp.set_start_method('spawn', force=True)

# 设置 GPU 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练参数
DATA_PATH = "../data/processed"
MODEL_PATH = "../models/best_model.pth"
BATCH_SIZE = 32  # batch_size GPU的计算负载
EPOCHS = 50
LEARNING_RATE = 3e-4  # 降低学习率，提高稳定性

# 数据预处理 & 数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),  
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
def load_data():
    train_dataset = ImageFolder(root=DATA_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    num_classes = len(train_dataset.classes)
    print(f"发现 {num_classes} 个类别: {train_dataset.classes}")
    return train_loader, num_classes

# 加载预训练模型，并确保在 GPU
def create_model(num_classes):
    # 加载 EfficientNet-B3 预训练模型
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)  # 确保模型在 GPU 上运行
    print(f"模型已加载到: {DEVICE}")

    # 冻结前几层，只训练分类层
    for param in model.parameters():
        param.requires_grad = False
   
    # 只冻结前 50% 的层，解冻后面的层
    #for name, param in model.named_parameters():
    #   if "layer4" in name or "fc" in name:  # 只训练最后几层
    #       param.requires_grad = True
    #   else:
    #       param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.classifier[1] = model.classifier[1].to(DEVICE)

    return model

#训练函数
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs):
    best_loss = float('inf')
    history = {'loss': [], 'accuracy': []}
    
    #添加 AMP 训练的 GradScaler
    scaler = torch.amp.GradScaler(device="cuda")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # 确保数据在 GPU

            optimizer.zero_grad()
    
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 进行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=running_loss / total, acc=correct / total)

        #计算 Epoch 平均 Loss & Accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

        #调整学习率
        scheduler.step(epoch_loss)

        # 保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"新最佳模型已保存: {MODEL_PATH}")
    
    return history

# 训练模型
if __name__ == "__main__":
    # 检查 GPU 可用性
    if torch.cuda.is_available():
        print(f"发现 GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.current_device()}")
    else:
        print("未找到 GPU，将使用 CPU 进行训练！")
        print("开始训练")

    # 加载数据
    train_loader, num_classes = load_data()

    # 加载模型
    model = create_model(num_classes)

    # 定义损失函数 & 优化器
    criterion = nn.CrossEntropyLoss(label_smoothing =0.1)
    optimizer = optim.AdamW(model.classifier[1].parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 开始训练
    start_time = time.time()
    history = train_model(model, train_loader, criterion, optimizer, scheduler, EPOCHS)
    print(f"训练完成，总耗时: {time.time() - start_time:.2f} 秒")

    # 画出 Loss & Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label="Loss")
    plt.plot(history['accuracy'], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Loss & Accuracy")
    plt.savefig("../models/training_curve.png")
    plt.show()
