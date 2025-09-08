#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"

# 필수/선택 환경변수 (없으면 기본값 사용) - 정적 빌드 전용
: "${MODEL_ONNX:=${MODEL_DIR}/model.onnx}"
: "${ENGINE_OUT:=${MODEL_DIR}/model_fp16.engine}"
: "${USE_FP16:=1}"          # 1이면 --fp16

# trtexec 경로 탐색(환경변수 우선, 없으면 자동 탐색)
TRTEXEC_BIN="${TRTEXEC:-$(command -v trtexec || true)}"
[ -z "$TRTEXEC_BIN" ] && [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
if [ ! -x "${TRTEXEC_BIN}" ]; then
  echo "trtexec not found. Install TensorRT or set TRTEXEC=<path-to-trtexec>." >&2
  exit 1
fi

# FP16 플래그
FPFLAG=""
[ "$USE_FP16" = "1" ] && FPFLAG="--fp16"

# 입력/출력 경로 절대화 및 모델 디렉터리 생성
mkdir -p "${MODEL_DIR}"
ONNX_ABS="$(realpath "$MODEL_ONNX")"
ENGINE_ABS="$(realpath -m "$ENGINE_OUT")"

echo "Using ONNX:   $ONNX_ABS"
echo "Saving engine: $ENGINE_ABS"
echo "trtexec bin : $TRTEXEC_BIN"

"${TRTEXEC_BIN}" \
  --onnx="$ONNX_ABS" \
  --saveEngine="$ENGINE_ABS" \
  $FPFLAG

echo "Done. Engine at: $ENGINE_ABS"