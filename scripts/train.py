#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py
PyTorchの標準的なImageFolderを使って、train/testのフォルダを分類学習するデモスクリプト。
進捗状況を表示するためにtqdmを利用しています。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # 進捗表示用

TRAIN_DIR = os.path.join("data", "train")
TEST_DIR = os.path.join("data", "test")
BATCH_SIZE = 8
EPOCHS = 2  # デモなので短め

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ変換設定
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ImageFolderでデータセットを作成
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # クラス数
    num_classes = len(train_dataset.classes)
    print("Number of classes:", num_classes, train_dataset.classes)

    # ResNet18のプリトレーニング済みモデルを使用し、出力層を差し替え
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 学習ループ
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        # tqdmで進捗表示
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # バッチごとの最新損失値を表示
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}")

    # テスト
    model.eval()
    correct = 0
    total = 0
    pbar_test = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for images, labels in pbar_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 学習済みモデルの保存
    torch.save(model.state_dict(), "plant_classifier.pth")
    print("Model saved to plant_classifier.pth")

if __name__ == "__main__":
    main()
