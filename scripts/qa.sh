#!/usr/bin/env bash

set -e

QA_FILE_CT=3
NLOOPS=${NLOOPS:-10}
MODEL_GGUF="${MODEL_GGUF:-$HOME/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf}"
LOC="${LOC:-.}"
CONTEXT_LENGTH=${CONTEXT_LENGTH:-102400}

export CUDA_VISIBLE_DEVICES=1,0

LOOP_NUMBER_STRING() {
    printf "%04d" "$1"
}

cargo build --release --bin caw-cli --features llama

for i in $(seq 1 $NLOOPS); do
    LOOP_NUMBER=$(LOOP_NUMBER_STRING "$i")
    echo "=== LOOP $LOOP_NUMBER ==="
    for j in $(seq 1 $QA_FILE_CT); do
        ./target/release/caw-cli --adapter llama --model "$MODEL_GGUF" --dir "$LOC" --num-ctx $CONTEXT_LENGTH --save-prompt < "qa/$(LOOP_NUMBER_STRING "$j").txt" >> qa/log.txt || break
    done
    mv .caw{,$LOOP_NUMBER}
    # TODO: call claude to evaluate the answer and improve the codebase
    # TODO: commit any changes claude makes
done
