#!/usr/bin/env bash
# Print the CUDA device ordinal with the most free VRAM, for use as
# CUDA_VISIBLE_DEVICES. Prints nothing (empty) when no NVIDIA GPU is
# present, which forces the candle embedder onto CPU.
#
# Why this exists: CandleEmbeddingProvider hardcodes Device::new_cuda(0)
# and only falls back to CPU when CUDA is *absent* — never on an OOM. On
# this host GPU 0 (the 3090) is usually full of an Ollama model, so a naive
# run OOMs mid-batch. Pinning CUDA_VISIBLE_DEVICES to the freest GPU makes
# whatever the embedder calls "cuda:0" actually be the GPU with headroom.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  exit 0
fi

# memory.free is reported in MiB, one line per GPU, index-ordered.
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk '{ if ($1 > max) { max = $1; idx = NR - 1 } } END { if (max > 0) print idx }'
