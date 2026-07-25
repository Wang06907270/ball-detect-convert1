"""
数字分类器训练 — MobileNetV2 (8类)
===================================
数据集: dataset/ (由 generate_dataset.py 生成)
输出:   models/best.pt (PyTorch权重)
        models/best.onnx (ONNX模型)
架构:   MobileNetV2, 224×224 输入, 8类输出
归一化: [0,1] = /255 (与 MaixCAM MUD 的 mean=0 scale=1/255 一致)

使用:
  python train_classifier.py        # 训练
  python train_classifier.py export # 仅导出 ONNX
"""

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os, sys, time

# ============================================================
# 配置
# ============================================================
DATA_DIR    = "dataset"
NUM_CLASSES = 8            # 数字 1-8
IMG_SIZE    = 224
BATCH       = 16
EPOCHS      = 30
LR          = 0.001
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR   = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# 归一化: [0,1] 范围（对应 MaixCAM MUD mean=0 scale=1/255）
# 训练时不做复杂归一化，直接用 /255
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),  # → [0,1]
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
])

# ============================================================
# 加载数据
# ============================================================
print("=" * 50)
print("  Digit Classifier Training (MobileNetV2)")
print("=" * 50)
print(f"Device: {DEVICE}")
print(f"Data:   {DATA_DIR}")

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=TRAIN_TRANSFORM)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=VAL_TRANSFORM)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

print(f"Classes: {train_ds.classes}")
print(f"Train:   {len(train_ds)} images")
print(f"Val:     {len(val_ds)} images")

# ============================================================
# 构建模型
# ============================================================
model = models.mobilenet_v2(weights=None, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ============================================================
# 训练
# ============================================================
best_acc = 0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    train_loss, train_correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # ── Val ──
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_ds)
    val_acc   = val_correct   / len(val_ds)
    elapsed   = time.time() - t_start

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"train loss: {train_loss/len(train_loader):.3f} acc: {train_acc:.3f} | "
          f"val loss: {val_loss/len(val_loader):.3f} acc: {val_acc:.3f} | "
          f"{elapsed:.0f}s")

    scheduler.step()

    # ── Save best ──
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{MODEL_DIR}/best.pt")
        print(f"  >> saved (val_acc={val_acc:.4f})")

print(f"\nTraining done. Best val_acc: {best_acc:.4f}")

# ============================================================
# 导出 ONNX
# ============================================================
print("\nExporting ONNX...")
model.load_state_dict(torch.load(f"{MODEL_DIR}/best.pt", map_location=DEVICE, weights_only=True))
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

torch.onnx.export(
    model, dummy,
    f"{MODEL_DIR}/best.onnx",
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12,
    dynamo=False,           # old-style exporter for tpu_mlir compat
)

# 验证 ONNX
import onnx
onnx_model = onnx.load(f"{MODEL_DIR}/best.onnx")
onnx.checker.check_model(onnx_model)
print(f"  ONNX OK: {MODEL_DIR}/best.onnx")
print(f"  Model size: {os.path.getsize(f'{MODEL_DIR}/best.onnx') / 1e6:.1f} MB")
print(f"\nNext: ONNX → CVIModel via GitHub Actions")
