#!/usr/bin/env bash
# Throughput baseline: run caw-bench-build-index across a small config
# matrix on opencaw-corpora/sysdoc and capture the stderr of each run.
# --rebuild on each run so resumes don't contaminate the numbers.
#
# Resumable: if OUTDIR is set to an existing directory, any config whose
# index-${tag}.sqlite already exists is skipped (log is preserved). Delete
# just the index file(s) for configs you want re-run. Without OUTDIR a
# fresh timestamped directory is created and every config runs.
set -euo pipefail

CORPUS="${CORPUS:-opencaw-corpora/sysdoc}"
if [[ -n "${OUTDIR:-}" ]]; then
    if [[ ! -d "${OUTDIR}" ]]; then
        echo "OUTDIR=${OUTDIR} is set but does not exist" >&2
        exit 1
    fi
    echo "resuming into existing ${OUTDIR}" >&2
else
    STAMP="$(date +%Y%m%d-%H%M%S)"
    OUTDIR="bench-results/throughput/${STAMP}"
    mkdir -p "${OUTDIR}"
fi
BIN="target/release/caw-bench-build-index"

if [[ ! -x "${BIN}" ]]; then
    echo "missing ${BIN}; run: cargo build --release -p caw-bench --bin caw-bench-build-index" >&2
    exit 1
fi

run() {
    local tag="$1" backend="$2" bs="$3" sb="$4"
    local idx="${OUTDIR}/index-${tag}.sqlite"
    local log="${OUTDIR}/${tag}.log"
    if [[ -f "${idx}" ]]; then
        echo "=== ${tag}: skip (index exists at ${idx})" | tee -a "${OUTDIR}/summary.txt"
        return
    fi
    # Stale WAL/SHM from an interrupted prior run would get picked up by
    # sqlite on open and either corrupt numbers or fail the rebuild. Clear
    # them whenever the main file is gone.
    rm -f "${idx}-wal" "${idx}-shm"
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
    {
        echo "--- final ---"
        grep -E "^done:|breakdown:" "${log}" || tail -4 "${log}"
        echo
    } | tee -a "${OUTDIR}/summary.txt"
}

if [[ -z "${NO_CANDLE:-}" ]]; then
    run candle-512-256 candle 512 256
    run candle-128-64  candle 128 64
fi
if [[ -z "${NO_ONNX:-}" ]]; then
    run onnx-512-256   onnx   512 256
    run onnx-128-64    onnx   128 64
fi

echo
echo "results: ${OUTDIR}/summary.txt"
