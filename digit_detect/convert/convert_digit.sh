#!/bin/bash
# ============================================================
# 数字分类 ONNX → CVIModel 转换 (Docker / GitHub Actions)
# ============================================================
# 用法:
#   Docker:
#     docker pull sophgo/tpuc_dev:latest
#     docker run --privileged -v $(pwd):/workspace -it sophgo/tpuc_dev:latest
#     容器内: bash convert_digit.sh
#
#   GitHub Actions: .github/workflows/convert-digit-model.yml
# ============================================================

set -e

MODEL_NAME="digit_cls"
ONNX_FILE="${1:-models/best.onnx}"
INPUT_SHAPE="${2:-[[1,3,224,224]]}"
PROCESSOR="${3:-cv181x}"
CALIB_DIR="${4:-calib_images}"
OUTPUT="${MODEL_NAME}_int8.cvimodel"

echo "=== Step 1: ONNX → MLIR ==="
model_transform.py \
    --model_name ${MODEL_NAME} \
    --model_def ${ONNX_FILE} \
    --input_shapes ${INPUT_SHAPE} \
    --pixel_format rgb \
    --mean 0,0,0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --mlir ${MODEL_NAME}.mlir

echo "=== Step 2: INT8 Calibration ==="
run_calibration.py \
    ${MODEL_NAME}.mlir \
    --dataset ${CALIB_DIR} \
    --input_num 50 \
    -o ${MODEL_NAME}_cali_table

echo "=== Step 3: MLIR → CVIModel ==="
model_deploy.py \
    --mlir ${MODEL_NAME}.mlir \
    --quantize INT8 \
    --calibration_table ${MODEL_NAME}_cali_table \
    --processor ${PROCESSOR} \
    --model ${OUTPUT}

echo "=== Done: ${OUTPUT} ==="
ls -lh ${OUTPUT}
