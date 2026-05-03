#!/usr/bin/env bash
set -euo pipefail

MAX_GB="${MAX_GB:-6}"
LIMIT="${LIMIT:-0}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OUTDIR="${OUTDIR:-}"
OUTFILE="${OUTFILE:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
# Context window cap passed as num_ctx to the Ollama API. Classification prompts
# fit in ~2k tokens; capping here prevents 32k-default models from allocating a
# giant KV cache and blowing VRAM when running many candidates back to back.
# Set to empty string to use each model's own default.
NUM_CTX="${NUM_CTX:-4096}"
RELEASE_FLAG="${RELEASE_FLAG:---release}"
NAME_REGEX="${NAME_REGEX:-}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "missing ollama in PATH" >&2
    exit 1
fi

if [[ -n "${OUTDIR}" && ! -d "${OUTDIR}" ]]; then
    echo "OUTDIR=${OUTDIR} is set but does not exist" >&2
    exit 1
fi

discover_models() {
    local lines
    lines="$({
        ollama list | awk -v max_gb="${MAX_GB}" -v name_regex="${NAME_REGEX}" '
        function size_to_gb(value, unit) {
            if (unit == "GB") return value + 0.0;
            if (unit == "MB") return (value + 0.0) / 1024.0;
            if (unit == "KB") return (value + 0.0) / (1024.0 * 1024.0);
            if (unit == "B") return (value + 0.0) / (1024.0 * 1024.0 * 1024.0);
            return -1.0;
        }

        NR == 1 { next }

        {
            name = $1;
            id = $2;
            size_value = $3;
            size_unit = $4;

            if (size_value == "-" || id == "") next;
            if (name_regex != "" && name !~ name_regex) next;

            size_gb = size_to_gb(size_value, size_unit);
            if (size_gb < 0 || size_gb > max_gb) next;

            if (!(id in best_name) || length(name) < length(best_name[id]) || (length(name) == length(best_name[id]) && name < best_name[id])) {
                best_name[id] = name;
                best_size[id] = size_gb;
            }
        }

        END {
            for (id in best_name) {
                printf "%.6f\t%s\n", best_size[id], best_name[id];
            }
        }
        ' | sort -n -k1,1
    } )"

    if [[ -z "${lines}" ]]; then
        return 0
    fi

    printf '%s\n' "${lines}" | cut -f2
}

model_supports_completion() {
    local model="$1"
    ollama show "${model}" 2>/dev/null | awk '
        /^[[:space:]]*Capabilities[[:space:]]*$/ { in_capabilities = 1; next }
        in_capabilities && /^[[:space:]]*Parameters[[:space:]]*$/ { exit }
        in_capabilities && $1 == "completion" { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

models=()
if [[ "$#" -gt 0 ]]; then
    for model in "$@"; do
        models+=("${model}")
    done
else
    while IFS= read -r model; do
        [[ -n "${model}" ]] || continue
        # if model contains 'prompter' 'aseio', 'moondream', or 'tinyllama', ignore it
        [[ "${model}" == *prompter* ]] || [[ "${model}" == *aseio* ]] || [[ "${model}" == *moondream* ]] || [[ "${model}" == *tinyllama* ]] && continue

        if model_supports_completion "${model}"; then
            models+=("${model}")
        else
            echo "skipping non-completion model ${model}" >&2
        fi
        
    done < <(discover_models)
fi

if [[ "${LIMIT}" -gt 0 && "${#models[@]}" -gt "${LIMIT}" ]]; then
    models=("${models[@]:0:${LIMIT}}")
fi

if [[ "${#models[@]}" -eq 0 ]]; then
    echo "no local Ollama models matched MAX_GB=${MAX_GB}${NAME_REGEX:+ NAME_REGEX=${NAME_REGEX}}" >&2
    exit 1
fi

if [[ -z "${OUTFILE}" ]]; then
    if [[ -z "${OUTDIR}" ]]; then
        stamp="$(date +%Y%m%d-%H%M%S)"
        OUTDIR="bench-results/intent/${stamp}"
        mkdir -p "${OUTDIR}"
    fi
    OUTFILE="${OUTDIR}/intent-report.json"
fi

cmd=(cargo run -p caw-bench --bin caw-bench-intent "${RELEASE_FLAG}" --
    --adapter ollama
    --ollama-url "${OLLAMA_URL}"
    --temperature "${TEMPERATURE}"
    --out "${OUTFILE}")

if [[ -n "${NUM_CTX}" ]]; then
    cmd+=(--num-ctx "${NUM_CTX}")
fi

for model in "${models[@]}"; do
    cmd+=(--candidate "${model}")
done

# Ensemble of top-3 models runs by default (--ensemble 0 to disable).

echo "selected ${#models[@]} model(s):" >&2
for model in "${models[@]}"; do
    echo "  ${model}" >&2
done
echo "writing ${OUTFILE}" >&2

"${cmd[@]}"
