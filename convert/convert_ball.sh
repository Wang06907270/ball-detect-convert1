#!/bin/bash
# ============================================================
# 钢球检测 ONNX → CVIModel 转换 (Docker)
# ============================================================
# 用法:
#   1. 把 export.onnx 放到 D:\code\maixcam\convert\
#   2. 准备校准图片到 convert/calib_images/ (50-200张train图片)
#   3. Docker 运行此脚本
#
#   docker pull sophgo/tpuc_dev:latest
#   docker run --privileged -v D:/code/maixcam/convert:/workspace -it sophgo/tpuc_dev:latest
#   容器内: bash convert_ball.sh
# ============================================================

set -e

MODEL_NAME="ball_detect"
ONNX_FILE="export.onnx"
CALIB_DIR="calib_images"
OUTPUT="${MODEL_NAME}_int8.cvimodel"

echo "=== Step 1: ONNX → MLIR ==="
model_transform.py \
    --model_name ${MODEL_NAME} \
    --model_def ${ONNX_FILE} \
    --input_shapes [[1,3,224,320]] \
    --pixel_format RGB \
    --mean 0,0,0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --mlir ${MODEL_NAME}.mlir

echo "=== Step 2: INT8 校准 ==="
run_calibration.py \
    ${MODEL_NAME}.mlir \
    --dataset ${CALIB_DIR} \
    --input_num 100 \
    -o ${MODEL_NAME}_cali_table

echo "=== Step 3: MLIR → CVIModel ==="
model_deploy.py \
    --mlir ${MODEL_NAME}.mlir \
    --quantize INT8 \
    --calibration_table ${MODEL_NAME}_cali_table \
    --processor cv181x \
    --model ${OUTPUT}

echo "=== Done: ${OUTPUT} ==="
echo "放到 D:\\code\\maixcam\\convert\\cvimodel\\"
echo "和 ball_detect.mud 一起上传到 MaixCAM /root/models/"
