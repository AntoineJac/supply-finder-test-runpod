#!/bin/bash
# start.sh — RunPod entrypoint
# 1. Launches SGLang server (GLM-4.7-Flash) in background
# 2. Launches FastAPI (embed + rerank + proxy) in foreground
set -euo pipefail

# ── Config (override via RunPod env vars) ─────────────────────────────────────
# Use pre-quantized FP8 model by default — fits on 24GB (L4, A5000, 3090)
# Switch to zai-org/GLM-4.7-Flash if you have 40GB+ (A100)
MODEL_PATH="${GLM_MODEL_PATH:-unsloth/GLM-4.7-Flash-FP8-Dynamic}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
API_PORT="${API_PORT:-8000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION_STATIC:-0.88}"   # 0.88 on 24GB, lower to 0.82 on 40GB
MAX_TOKENS="${MAX_TOTAL_TOKENS:-32768}"        # cap context to save KV cache VRAM
WORKERS="${UVICORN_WORKERS:-2}"

# HuggingFace cache — RunPod persists /workspace between runs
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

echo "============================================================"
echo "  NLP Inference Service"
echo "  GLM model   : ${MODEL_PATH}"
echo "  SGLang      : port ${SGLANG_PORT}  TP=${TP_SIZE}"
echo "  FastAPI     : port ${API_PORT}  workers=${WORKERS}"
echo "  Max tokens  : ${MAX_TOKENS}"
echo "  Mem fraction: ${MEM_FRACTION}"
echo "  HF cache    : ${HF_HOME}"
echo "============================================================"

# ── 1. Start SGLang ───────────────────────────────────────────────────────────
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${SGLANG_PORT}" \
  --tp-size "${TP_SIZE}" \
  --dtype bfloat16 \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-total-tokens "${MAX_TOKENS}" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --served-model-name glm-4.7-flash \
  --trust-remote-code \
  &

SGLANG_PID=$!
echo "SGLang started (PID ${SGLANG_PID})"

# ── 2. Wait for SGLang to be healthy ─────────────────────────────────────────
echo "Waiting for SGLang on port ${SGLANG_PORT}..."
SGLANG_READY=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
    echo "SGLang is healthy after ${i}×5s"
    SGLANG_READY=1
    break
  fi
  sleep 5
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "ERROR: SGLang process died. Check logs above."
    exit 1
  fi
done

if [ "${SGLANG_READY}" -eq 0 ]; then
  echo "ERROR: SGLang failed to become healthy within 5 minutes."
  kill "${SGLANG_PID}" 2>/dev/null || true
  exit 1
fi

# ── 3. Start FastAPI ──────────────────────────────────────────────────────────
export SGLANG_BASE_URL="http://localhost:${SGLANG_PORT}"

exec uvicorn app:app \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --workers "${WORKERS}" \
  --log-level info \
  --no-access-log