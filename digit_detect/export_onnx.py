"""
导出 PyTorch 模型为 ONNX (用于 MaixCAM 部署)
=============================================
从 models/best.pt 导出 → models/best.onnx
使用: python export_onnx.py (需先 train_classifier.py 训练完成)
"""

import torch, onnx, os, sys
from torchvision import models

IMG_SIZE = 224
NUM_CLASSES = 8
DEVICE = "cpu"
MODEL_DIR = "models"

print("Loading model...")
model = models.mobilenet_v2(weights=None, num_classes=NUM_CLASSES)
state = torch.load(f"{MODEL_DIR}/best.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

print("Exporting ONNX...")
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model, dummy,
    f"{MODEL_DIR}/best.onnx",
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12,
    dynamo=False,           # use old-style exporter (compatible with tpu_mlir)
)

onnx_model = onnx.load(f"{MODEL_DIR}/best.onnx")
onnx.checker.check_model(onnx_model)
size_mb = os.path.getsize(f"{MODEL_DIR}/best.onnx") / 1e6
print(f"Done: {MODEL_DIR}/best.onnx ({size_mb:.1f} MB)")
