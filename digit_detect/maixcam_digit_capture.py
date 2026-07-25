"""
MaixCAM 数字拍照采集工具
========================
功能: 触屏点击拍照，用于采集数字分类训练数据
      照片存到 /tmp/digit_photos/

使用:
  1. MaixVision 打开本文件 → 运行
  2. 打印 数字.pdf，把镜头对准数字卡片
  3. 点击屏幕拍照
  4. 多样角度、距离、光照
  5. 建议每个数字拍 15-20 张

注意:
  - 数字卡片尽量占满画面
  - 覆盖不同倾斜角度和距离
"""

from maix import camera, display, image, app, touchscreen
import time, os, cv2

CAM_WIDTH, CAM_HEIGHT = 320, 240
SAVE_DIR = "/tmp/digit_photos"

print("=" * 50)
print("  Digit Photo Capture")
print("=" * 50)

os.makedirs(SAVE_DIR, exist_ok=True)
existing = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
count = len(existing)
print(f"  Existing: {count}")

cam = camera.Camera(CAM_WIDTH, CAM_HEIGHT, image.Format.FMT_RGB888, fps=60)
cam.skip_frames(30)
print(f"  Camera {CAM_WIDTH}x{CAM_HEIGHT} ready")

disp = display.Display()
ts = touchscreen.TouchScreen()

last_touch = False
flash_timer = 0

while not app.need_exit():
    img = cam.read()
    if img is None:
        continue

    frame = image.image2cv(img, ensure_bgr=True, copy=False)

    # 提示
    cv2.putText(frame, f"Saved: {count}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Tap to capture", (8, CAM_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    if flash_timer > 0:
        flash_timer -= 1
        cv2.circle(frame, (CAM_WIDTH // 2, CAM_HEIGHT // 2),
                   35, (0, 255, 255), 3)

    touch = ts.read()
    touched = touch[2]

    if touched and not last_touch:
        count += 1
        fname = f"digit_{count:04d}.jpg"
        img.save(os.path.join(SAVE_DIR, fname))
        print(f"  [{count}] {fname}")
        flash_timer = 8

    last_touch = touched

    disp.show(image.cv2image(frame, bgr=True, copy=False))

    if os.path.exists("/tmp/stop_cam"):
        os.remove("/tmp/stop_cam")
        break

cam.close()
print(f"Done. Total: {count}")
