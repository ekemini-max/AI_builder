import os
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from modelforge.dataset.splitter import stratified_data_split


def get_vision_transforms():
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def train_vision_model(
    image_paths: List[str],
    class_labels: List[str],
    output_model_path: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3
) -> Dict[str, Any]:
    """
    Trains a lightweight MobileNetV2 classification model on CPU using transfer learning.
    Imports PyTorch and Torchvision lazily to prevent startup memory overhead.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models

    class ImageDataset(Dataset):
        def __init__(self, image_paths: List[str], labels: List[int], transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            label = self.labels[idx]
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), (0, 0, 0))

            if self.transform:
                image = self.transform(image)

            return image, label

    unique_classes = sorted(list(set(class_labels)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(unique_classes)}
    numeric_labels = [class_to_idx[cls] for cls in class_labels]

    tr_paths, tr_labels, val_paths, val_labels, te_paths, te_labels = stratified_data_split(
        image_paths, numeric_labels, test_size=0.2, val_size=0.1
    )

    train_tf, eval_tf = get_vision_transforms()

    train_ds = ImageDataset(tr_paths, tr_labels, transform=train_tf)
    val_ds = ImageDataset(val_paths, val_labels, transform=eval_tf)
    test_ds = ImageDataset(te_paths, te_labels, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(unique_classes)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    device = torch.device("cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    per_class_metrics = {}
    for idx, cls_name in idx_to_class.items():
        per_class_metrics[cls_name] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx])
        }

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": unique_classes,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        "model_architecture": "mobilenet_v2"
    }, output_model_path)

    return {
        "eval_split_type": "held_out_test",
        "is_trustworthy_held_out": True,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(np.mean(precision)), 4),
        "macro_recall": round(float(np.mean(recall)), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_mat,
        "classes": unique_classes,
        "format": "pt"
    }
