# MaixCAM 模型转换

## 已有模型

| 文件 | 说明 |
|------|------|
| `cvimodel/target_detect_int8.cvimodel` (3.2MB) | 靶子检测 ✅ 已完成 |
| `target_detect.mud` | 靶子检测配置文件 |
| `export.onnx` (12MB) | 靶子检测 ONNX（参考） |
| `ball_detect.mud` | 钢球检测 MUD 模板 ✅ 已准备 |
| `convert_ball.sh` | 钢球检测 Docker 转换脚本 ✅ 已准备 |

## 下一步: 钢球检测 ONNX → CVIModel

### ✅ 前提已完成
1. ✅ 训练完成 — 本地 RTX 4060，YOLOv8n, 100 epochs, imgsz=320
2. ✅ `ball_export.onnx` (12MB) — 已裁剪2个TPU输出 + onnxsim简化
3. ✅ `calib_images/` (100张) — 校准图片已就绪
4. ✅ GitHub Actions 工作流 — `.github/workflows/convert-ball-model.yml`

### ⏳ 转换: GitHub Actions

1. 在 GitHub 创建空仓库（如 `maixcam-models`）
2. 推送到 GitHub:
   ```bash
   cd D:\code\maixcam
   git init
   git add convert/ .github/workflows/ .gitignore
   git commit -m "ball detection ONNX → CVIModel via GitHub Actions"
   git remote add origin https://github.com/YOUR_USER/maixcam-models.git
   git push -u origin main
   ```
3. GitHub Actions 自动运行 → 下载 `ball_detect_int8.cvimodel` (Artifacts)
4. 放到 `convert/cvimodel/`

### 输出

转换成功后得到 `ball_detect_int8.cvimodel`，放到 `convert/cvimodel/`。

## 部署到 MaixCAM (192.168.1.139)

```bash
# 上传模型
scp ball_detect.mud ball_detect_int8.cvimodel root@192.168.1.139:/root/models/

# 运行检测
# MaixVision 打开 maixcam_ball_yolo.py → 运行
```
