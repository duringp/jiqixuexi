import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image


class HandGestureDataset(Dataset):
    def __init__(self, data_dir='静态手势数据集/merged', label_file='data_file.txt'):
        self.image_label = []
        self.data_dir = data_dir
        
        # 检查标签文件是否存在
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.image_label = f.readlines()
            print(f"从 {label_file} 加载了 {len(self.image_label)} 个样本")
        else:
            # 如果没有标签文件，直接扫描目录
            print(f"标签文件 {label_file} 不存在，扫描目录 {data_dir}")
            for filename in os.listdir(data_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_label.append(filename)
            print(f"从目录加载了 {len(self.image_label)} 个样本")
        
        # 过滤掉不存在的文件
        valid_samples = []
        for image_file in self.image_label:
            image_file = image_file.replace('\n', '')
            file_path = os.path.join(self.data_dir, image_file)
            if os.path.exists(file_path):
                valid_samples.append(image_file)
        
        self.image_label = valid_samples
        print(f"有效样本数: {len(self.image_label)}")

    def __getitem__(self, index):
        image_file = self.image_label[index]
        image_file = image_file.replace('\n', '')
        
        # 从文件名提取标签 (格式: XX-YYYYYY.jpg)
        if '-' in image_file:
            label = int(image_file.split('-')[0])
        else:
            # 如果文件名格式不对，使用默认标签
            label = 0

        file_name = os.path.join(self.data_dir, image_file)

        # 使用PIL读取图像（支持中文路径）
        image = Image.open(file_name).convert('RGB')
        image = np.array(image)
        
        # 调整图像大小为32x32
        image = cv2.resize(image, (32, 32))

        # 转换为张量并归一化到[0,1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, label

    def __len__(self):
        return len(self.image_label)


# 定义 CNN 模型
class HandGestureCNN(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        # 经过2次池化后，32x32 -> 8x8
        # 64通道 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 第一层卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # 第二层卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # 展平
        x = x.view(len(x), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x


# ===================== GPU 配置 =====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"训练使用设备: {device}")
# ====================================================

# 加载数据集
full_dataset = HandGestureDataset()

# 划分训练集和测试集 (80%训练, 20%测试)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 模型、损失、优化器
model = HandGestureCNN(num_classes=18).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型（带进度条）
def train(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # ========== 加了进度条 ==========
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 实时显示当前损失
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / len(train_loader)
        print(f"===> Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")


# 测试模型
def do_test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    print("\n正在测试...")
    with torch.no_grad():
        # 测试也加进度条
        for data, target in tqdm(test_loader, desc="测试进度"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'\n测试准确率: {100 * correct / total:.2f}%')


def save_model(model):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_file = './models/HandGestureCNN.pth'
    torch.save(model.state_dict(), model_file)
    print(f"模型已保存: {model_file}")


if __name__ == '__main__':
    train(model, train_loader, criterion, optimizer, epochs=30, device=device)
    do_test(model, test_loader, device)
    save_model(model)