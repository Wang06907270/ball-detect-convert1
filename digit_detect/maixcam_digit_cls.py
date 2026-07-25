"""
MaixCAM 数字识别 v1 — MobileNetV2 分类器
=========================================
功能:
  1. MobileNetV2 识别数字 1-8
  2. 屏幕显示识别结果 + 置信度
  3. 通过 UART 串口发送识别到的数字

使用:
  1. 上传模型到 /tmp/ (digit_cls.mud + digit_cls_int8.cvimodel)
  2. MaixVision 打开本脚本 → 运行
  3. 将镜头对准打印的数字卡片
"""

from maix import camera, display, image, app, touchscreen, nn, uart
import time, os, cv2

# ============================================================
# 配置
# ============================================================
MODEL_DIR   = "/tmp"
MUD_NAME    = "digit_cls.mud"

CAM_WIDTH, CAM_HEIGHT = 320, 240
DISP_WIDTH, DISP_HEIGHT = 640, 480   # 显示分辨率

# ── 分类 ──
CONF_TH = 0.6   # 低于此置信度显示 "?"

# ── 串口 ──
UART_PORT   = "/dev/ttyS0"
UART_BAUD   = 115200
UART_ENABLE = True

# ============================================================
# 初始化
# ============================================================
print("=" * 50)
print("  MaixCAM Digit Recognizer v1")
print("=" * 50)

MODEL_PATH = os.path.join(MODEL_DIR, MUD_NAME)
print(f"Model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"\n[E] {MUD_NAME} not found in {MODEL_DIR}/")
    print(f"  Upload: digit_cls.mud + digit_cls_int8.cvimodel → {MODEL_DIR}/")
    raise FileNotFoundError(MODEL_PATH)

print("Loading classifier...")
classifier = nn.Classifier(model=MODEL_PATH)
print("  model OK")

# ── 串口 ──
uart_dev = None
if UART_ENABLE:
    try:
        uart_dev = uart.UART(UART_PORT, UART_BAUD)
        print(f"UART:  {UART_PORT} @ {UART_BAUD}")
    except Exception as e:
        print(f"[W] UART failed: {e}")

def uart_send(msg):
    if uart_dev is None:
        return
    try:
        uart_dev.write(msg + "\r\n")
    except Exception:
        pass

# ── 摄像头 ──
print("Opening camera...")
cam = camera.Camera(CAM_WIDTH, CAM_HEIGHT, image.Format.FMT_RGB888, fps=60)
cam.skip_frames(30)
print(f"  {CAM_WIDTH}x{CAM_HEIGHT} ready")

disp = display.Display()
ts = touchscreen.TouchScreen()

# ============================================================
# 主循环
# ============================================================
t0, n, fps = time.time(), 0, 0
last_digit = -1

print("\nPoint camera at a digit (1-8). Touch screen to exit.\n")

while not app.need_exit():
    img = cam.read()
    if img is None:
        continue

    # ── 分类 ──
    result = classifier.classify(img)
    # result 格式: [{"class_id": 0, "score": 0.95}, ...] 或者直接返回 class_id

    # nn.Classifier.classify 返回 list[dict]
    if isinstance(result, list) and len(result) > 0:
        top = result[0]
        cls_id = top.get("class_id", top.get("id", 0))
        score  = top.get("score", top.get("prob", 0.0))
    else:
        cls_id = result if isinstance(result, int) else 0
        score  = 1.0

    digit = cls_id + 1          # class 0 = digit 1, class 1 = digit 2 ...
    confident = score >= CONF_TH

    # ── 串口发送 ──
    if confident and digit != last_digit:
        uart_send(f"DIGIT:{digit}")
        print(f"  UART >>> DIGIT:{digit}  (conf={score:.2f})")
        last_digit = digit
    elif not confident and last_digit != -1:
        uart_send("DIGIT:?")
        last_digit = -1

    # ── 绘制 ──
    frame = image.image2cv(img, ensure_bgr=True, copy=False)
    frame = cv2.resize(frame, (DISP_WIDTH, DISP_HEIGHT))

    # 背景
    cv2.rectangle(frame, (0, 0), (DISP_WIDTH, DISP_HEIGHT), (20, 20, 20), -1)

    # 识别结果大字居中
    if confident:
        text = str(digit)
        color = (0, 255, 0)
    else:
        text = "?"
        color = (100, 100, 100)

    # 大字
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)[0]
    cx, cy = (DISP_WIDTH - sz[0]) // 2, (DISP_HEIGHT + sz[1]) // 2
    cv2.putText(frame, text, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 6, color, 8)

    # 置信度条
    if confident:
        bar_w = int(score * (DISP_WIDTH - 60))
        cv2.rectangle(frame, (30, DISP_HEIGHT - 40), (DISP_WIDTH - 30, DISP_HEIGHT - 15),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (30, DISP_HEIGHT - 40), (30 + bar_w, DISP_HEIGHT - 15),
                      (0, 255, 0), -1)
        cv2.putText(frame, f"{score:.0%}", (DISP_WIDTH // 2 - 30, DISP_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 顶部标签
    label = "Digit Recognizer v1"
    cv2.putText(frame, label, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    if uart_dev:
        cv2.putText(frame, "TX", (DISP_WIDTH - 30, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    # ── 显示 ──
    disp.show(image.cv2image(frame, bgr=True, copy=False))

    # ── FPS ──
    n += 1
    if n >= 10:
        dt = time.time() - t0
        fps = int(n / max(dt, 0.001))
        t0, n = time.time(), 0

    # ── 退出 ──
    if ts.read()[2]:
        break
    if os.path.exists("/tmp/stop_cam"):
        os.remove("/tmp/stop_cam")
        break

cam.close()
print("Done.")
