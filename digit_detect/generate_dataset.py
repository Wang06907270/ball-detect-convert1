"""
数字分类器 — 数据增强脚本
==========================
从8张PDF导出的数字图片 (digit_1.png ~ digit_8.png)
通过数据增强生成训练集和验证集：
  train: ~25张/类 × 8类 = 200张
  val:   ~5张/类  × 8类 = 40张

增强策略：旋转、缩放、平移、亮度、噪声、透视
输出结构 (ImageFolder 格式):
  dataset/
    train/
      1/  (digit_1_000.jpg ...)
      2/
      ...
      8/
    val/
      1/
      ...
      8/
"""

import cv2, os, random, numpy as np

SRC_DIR   = "images"
OUT_DIR   = "dataset"
PER_CLASS = 25           # 每类生成数量
VAL_RATIO = 0.2          # 验证集比例
IMG_SIZE  = 224          # MobileNetV2 标准输入

os.makedirs(OUT_DIR, exist_ok=True)

def load_digit(n):
    """加载数字图片，裁剪白边，缩放到统一尺寸"""
    path = f"{SRC_DIR}/digit_{n}.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [E] not found: {path}")
        return None

    # 二值化 + 找轮廓裁剪白边
    _, th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    # 留一点边距
    pad = 10
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(img.shape[1] - x, w + 2 * pad), min(img.shape[0] - y, h + 2 * pad)
    crop = img[y:y+h, x:x+w]

    # 等比缩放留边 (letterbox)
    scale = IMG_SIZE / max(crop.shape)
    new_w, new_h = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
    resized = cv2.resize(crop, (new_w, new_h))

    canvas = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    dx, dy = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    return canvas


def augment(img):
    """随机增强，返回 BGR 图像"""
    h, w = img.shape

    # 1) 旋转
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # 2) 缩放（<=1 缩小后居中，>1 放大后居中裁剪）
    scale = random.uniform(0.7, 1.3)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(img, (new_w, new_h))
    if new_w <= IMG_SIZE and new_h <= IMG_SIZE:
        canvas = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        dx = (IMG_SIZE - new_w) // 2 + random.randint(-5, 5)
        dy = (IMG_SIZE - new_h) // 2 + random.randint(-5, 5)
        dx = max(0, min(dx, IMG_SIZE - new_w))
        dy = max(0, min(dy, IMG_SIZE - new_h))
        canvas[dy:dy+new_h, dx:dx+new_w] = scaled
        img = canvas
    else:
        # 放大 → 居中裁剪回 IMG_SIZE
        cx, cy = new_w // 2, new_h // 2
        hw = IMG_SIZE // 2
        img = scaled[cy-hw:cy+hw, cx-hw:cx+hw]

    # 3) 透视变换 (模拟拍摄角度)
    if random.random() < 0.4:
        margin = int(IMG_SIZE * 0.08)
        src = np.float32([[0,0], [IMG_SIZE,0], [0,IMG_SIZE], [IMG_SIZE,IMG_SIZE]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [IMG_SIZE - random.randint(0, margin), random.randint(0, margin)],
            [random.randint(0, margin), IMG_SIZE - random.randint(0, margin)],
            [IMG_SIZE - random.randint(0, margin), IMG_SIZE - random.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE), borderValue=255)

    # 4) 亮度/对比度
    alpha = random.uniform(0.7, 1.4)   # 对比度
    beta  = random.randint(-25, 25)    # 亮度
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 5) 高斯噪声
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 6) 轻微模糊
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 转 BGR (三通道，分类模型需要)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr


# ============================================================
# 主流程
# ============================================================
print("=" * 50)
print("  Digit Dataset Generator")
print("=" * 50)

random.seed(42)
np.random.seed(42)

for digit in range(1, 9):
    base = load_digit(digit)
    if base is None:
        continue

    n_val = max(1, int(PER_CLASS * VAL_RATIO))
    n_train = PER_CLASS - n_val

    for subset, count in [("train", n_train), ("val", n_val)]:
        subdir = f"{OUT_DIR}/{subset}/{digit}"
        os.makedirs(subdir, exist_ok=True)
        for i in range(count):
            aug = augment(base)
            fname = f"{subdir}/digit_{digit}_{i:03d}.jpg"
            cv2.imwrite(fname, aug)

    print(f"  digit_{digit}: {n_train} train + {n_val} val")

# 写 labels.txt (给 MaixVision 参考)
with open(f"{OUT_DIR}/labels.txt", "w") as f:
    for d in range(1, 9):
        f.write(f"{d}\n")

print(f"\nDone. Total: {PER_CLASS * 8} images")
print(f"  train: {OUT_DIR}/train/{{1..8}}/")
print(f"  val:   {OUT_DIR}/val/{{1..8}}/")
