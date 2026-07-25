# MaixCAM 数字识别 (Digit Classifier)

MobileNetV2 分类器，识别数字 **1-8**（打印体）。

## 项目结构

```
digit_detect/
├── README.md                      ← 本文件
├── images/                        ← PDF 导出的源图 (digit_1~8.png)
├── generate_dataset.py            ← 数据增强 → dataset/
├── train_classifier.py            ← 训练 MobileNetV2 分类器
├── export_onnx.py                 ← PT → ONNX
├── maixcam_digit_cls.py           ← ★ MaixCAM 主脚本 (MaixVision 运行)
├── maixcam_digit_capture.py       ← 拍照采集工具
├── models/
│   ├── digit_cls.mud              ← MUD 模型配置
│   ├── best.pt / best.onnx        ← 训练产物
│   └── digit_cls_int8.cvimodel    ← 转换后模型 (需手动下载)
├── dataset/                       ← 增强后训练数据 (auto-generated)
│   ├── train/{1..8}/
│   └── val/{1..8}/
└── convert/
    └── convert_digit.sh           ← Docker 转换脚本
```

## 快速开始

### 1. 生成数据集

```bash
python generate_dataset.py
# → 生成 ~200 张增强图片到 dataset/
```

### 2. 训练模型

```bash
python train_classifier.py
# → 输出 models/best.pt + models/best.onnx
```

### 3. 转换模型 (二选一)

**A) GitHub Actions (推荐，免费)**
1. 确保 `models/best.onnx` 已提交
2. 准备校准图片: 复制 50 张 `dataset/train/` 下的图到 `calib_images/`
3. Push → Actions 自动转换 → 下载 artifact

**B) Docker 本地**
```bash
bash convert/convert_digit.sh
```

### 4. 部署到 MaixCAM

将以下文件上传到 MaixCAM 的 `/tmp/`:
- `models/digit_cls.mud`
- `models/digit_cls_int8.cvimodel`

MaixVision 打开 `maixcam_digit_cls.py` → 运行

### 5. 采集真实数据 (可选)

如果增强数据效果不好，用 MaixCAM 直接采集:
1. 打印 `数字.pdf`
2. MaixVision 运行 `maixcam_digit_capture.py`
3. 点击屏幕拍照，每个数字拍 15-20 张
4. 人工标注: 按数字分到 `dataset/train/{1..8}/`
5. 重新训练

## 模型信息

| 项 | 值 |
|------|-----|
| 架构 | MobileNetV2 |
| 输入 | 224×224 RGB |
| 输出 | 8 类 (数字 1-8) |
| 归一化 | mean=0, scale=1/255 |
| 处理器 | cv181x |
| 类型 | classifier |

## 串口输出

默认启用 UART0 (TX=IO17)，波特率 115200:
```
DIGIT:3      ← 识别到数字 3
DIGIT:?      ← 置信度低，无法确定
```
