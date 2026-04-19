#!/usr/bin/env bash
# Throughput baseline: run caw-bench-build-index across a small config
# matrix on opencaw-corpora/sysdoc and capture the stderr of each run.
# --rebuild on each run so resumes don't contaminate the numbers.
set -euo pipefail

CORPUS="${CORPUS:-opencaw-corpora/sysdoc}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="bench-results/throughput/${STAMP}"
BIN="target/release/caw-bench-build-index"

mkdir -p "${OUTDIR}"

if [[ ! -x "${BIN}" ]]; then
    echo "missing ${BIN}; run: cargo build --release -p caw-bench --bin caw-bench-build-index" >&2
    exit 1
fi

run() {
    local tag="$1" backend="$2" bs="$3" sb="$4"
    local idx="${OUTDIR}/index-${tag}.sqlite"
    local log="${OUTDIR}/${tag}.log"
    echo "=== ${tag}: backend=${backend} batch=${bs} sub_batch=${sb}" | tee -a "${OUTDIR}/summary.txt"
    "${BIN}" \
        --corpus "${CORPUS}" \
        --out "${idx}" \
        --rebuild \
        --backend "${backend}" \
        --batch-size "${bs}" \
        --sub-batch-size "${sb}" \
        --log-interval 10 \
        2> "${log}"
    # Extract the final summary block (last three "breakdown" lines + the
    # "done:" line) into summary.txt for at-a-glance comparison.
    {
        echo "--- final ---"
        grep -E "^done:|breakdown:" "${log}" || tail -4 "${log}"
        echo
    } | tee -a "${OUTDIR}/summary.txt"
}

run candle-512-256 candle 512 256
run candle-128-64  candle 128 64
run onnx-512-256   onnx   512 256
run onnx-128-64    onnx   128 64

echo
echo "results: ${OUTDIR}/summary.txt"
