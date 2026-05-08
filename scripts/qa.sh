#!/usr/bin/env bash

set -euo pipefail

QA_FILE_CT=${QA_FILE_CT:-3}
NLOOPS=${NLOOPS:-10}
MODEL_GGUF="${MODEL_GGUF:-$HOME/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf}"
LOC="${LOC:-.}"
CONTEXT_LENGTH=${CONTEXT_LENGTH:-102400}
MAX_FIX_ATTEMPTS=${MAX_FIX_ATTEMPTS:-3}
REGENERATE=${REGENERATE:-1}

export CUDA_VISIBLE_DEVICES=1,0

# Kill the entire process group on Ctrl-C so claude subprocesses don't linger.
trap 'kill 0' INT TERM

LOOP_NUMBER_STRING() {
    printf "%04d" "$1"
}

LOOP_START() {
    # find the last .cawNNNN directory and extract the number, or return 0 if none found
    local last_loop=0
    for dir in .caw*/; do
        if [[ "$dir" =~ \.caw([0-9]{4})/ ]]; then
            num="${BASH_REMATCH[1]}"
            if (( 10#$num > last_loop )); then
                last_loop=$(( 10#$num ))
            fi
        fi
    done
    echo "$last_loop"
}

# Build caw-cli; on failure invoke Claude to fix errors and retry.
# Usage: build_or_fix <label>
build_or_fix() {
    local label="$1"
    local attempt=1
    while [ "$attempt" -le "$MAX_FIX_ATTEMPTS" ]; do
        echo "=== BUILD ($label, attempt $attempt/$MAX_FIX_ATTEMPTS) ==="
        if cargo build --release --bin caw-cli --features llama; then
            return 0
        fi

        echo "Build failed — launching Claude to fix compilation errors..."
        CLAUDE_BUGFIX_PROMPT=$(cat <<BUGFIX_PROMPT
The opencaw Rust codebase at $(pwd) does not compile.

opencaw is a workspace-aware LLM context augmentation system. Crate layout:
- caw-orchestrator: core retrieval, session, consolidation
- caw-adapters: LLM adapter implementations (llama, anthropic, save-prompt)
- caw-workspace: workspace indexing, stub generation
- caw-cli: CLI entry point
- caw-index, caw-ingest, caw-transform, caw-core, caw-curation, caw-eval, caw-bench, caw-server

Run: cargo build --release --bin caw-cli --features llama
Fix ONLY the compilation errors it reports. Do not refactor, add features, or touch unrelated code.
Do not stop until the build succeeds.
BUGFIX_PROMPT
        )
        claude --dangerously-skip-permissions -p "$CLAUDE_BUGFIX_PROMPT" \
            || echo "Claude failed to fix the build, but retrying build loop anyway."

        attempt=$((attempt + 1))
    done
    echo "=== BUILD FAILED after $MAX_FIX_ATTEMPTS attempts — aborting ==="
    return 1
}

mkdir -p qa/recommendations

build_or_fix "initial"

LOOP_OFFSET=$(LOOP_START)
END_LOOP=$((LOOP_OFFSET + NLOOPS))
for n in $(seq 1 "$NLOOPS"); do
    i=$((LOOP_OFFSET + n))
    LOOP_NUMBER=$(LOOP_NUMBER_STRING "$i")
    echo ""
    echo "========================================================"
    echo "=== OUTER LOOP $LOOP_NUMBER of $END_LOOP ==="
    echo "========================================================"

    # Run each QA prompt through caw-cli, accumulating session artifacts in .caw
    for j in $(seq 1 "$QA_FILE_CT"); do
        QA_FILE="qa/$(LOOP_NUMBER_STRING "$j").txt"
        if [ ! -f "$QA_FILE" ]; then
            echo "No $QA_FILE, skipping."
            continue
        fi
        echo "--- Generation: qa prompt $j ---"
        ./target/release/caw-cli \
            --adapter llama \
            --model "$MODEL_GGUF" \
            --dir "$LOC" \
            --num-ctx "$CONTEXT_LENGTH" \
            --save-prompt \
            < "$QA_FILE" \
            >> qa/log.txt || {
            echo "caw-cli failed on prompt $j, stopping inner loop."
            break
        }
    done

    mv .caw ".caw${LOOP_NUMBER}"
    if [ "${REGENERATE:-1}" -eq 1 ]; then
        echo "Regenerating index.db for loop $LOOP_NUMBER to ensure it reflects the final session artifacts..."
    else
        echo "NOTE: Skipping index.db regeneration for loop $LOOP_NUMBER; if session artifacts changed during generation, the index may be out of sync."
        mkdir -p .caw/
        cp ".caw${LOOP_NUMBER}/index.db" .caw/ || echo "No index.db found in .caw${LOOP_NUMBER}, skipping copy to .caw/"
    fi

    # ----------------------------------------------------------------
    # REVIEW PASS: Claude reads the session output and DB to assess
    # what the QA runs reveal about context quality and retrieval.
    # ----------------------------------------------------------------
    echo "--- Review pass (Claude) ---"
    CLAUDE_REVIEW_PROMPT=$(cat <<REVIEW_PROMPT
You are reviewing the output of QA sessions run by opencaw (loop $LOOP_NUMBER) to identify what can be learned about context quality and retrieval effectiveness.

WHAT OPENCAW DOES:
opencaw is a workspace-aware context augmentation system for LLMs. It retrieves relevant code summaries
(stubs) from a local index and injects them as context before each model query. The goal is to give the
model accurate, relevant workspace context so its answers are grounded and useful.

SESSION ARTIFACTS in .caw${LOOP_NUMBER}/:
- session-*.md files — the actual QA session logs: prompts sent to the model, context injected, and model responses
- prompt-*.txt files — the actual context sent to the model for each turn
- index.db — SQLite DB with the stubs that were available for retrieval:
    stubs(id, path, token_estimate, kind, summary, outline, ...)
    consolidation_notes(id, stub_id, content, source, created_at_secs)

YOUR TASK:
1. Read the session files in .caw${LOOP_NUMBER}/ to see what was asked, what context was injected, and what the model said
2. Query .caw${LOOP_NUMBER}/index.db to understand what stubs were available and whether the right ones were retrieved:
     SELECT path, kind, summary, token_estimate FROM stubs ORDER BY token_estimate DESC LIMIT 30;
     SELECT content, source FROM consolidation_notes LIMIT 10;
3. Write a numbered, prioritized list of observations to: qa/recommendations/${LOOP_NUMBER}.md
4. Add improvement recommendations to TODO.md, and add bugs found to BUGS.md

Evaluate on these dimensions:
- Were the model's answers accurate and grounded in the injected context?
- Did the retrieved stubs match what the queries needed? Any obvious misses or irrelevant inclusions?
- Was the injected context too verbose, noisy, or missing key signal?
- Did consolidation notes add useful cross-stub context, or were they redundant?
- Any patterns in what the model got wrong that point to retrieval or summarization problems?
- Every qa session should contain 3 turns. Are there any turns missing?
- The model should have all the information it needs to respond accurately in the injected context. Are there any cases where the model's response indicates it lacked necessary information that should have been retrieved?

Write observations about what the session output reveals — what is going wrong and why it matters.
You may look at the Rust source to understand why something behaves the way it does, but the session output is your primary evidence.
Each observation should be concrete enough that an implementer can act on it.
REVIEW_PROMPT
    )
    claude --dangerously-skip-permissions -p "$CLAUDE_REVIEW_PROMPT" \
        | tee qa/claude_review_${LOOP_NUMBER}.txt \
        || echo "Claude failed during review pass, but continuing to implementation."

    if [ ! -f "qa/recommendations/${LOOP_NUMBER}.md" ]; then
        echo "WARNING: Claude did not write qa/recommendations/${LOOP_NUMBER}.md — skipping implementation pass."
        continue
    fi

    # ----------------------------------------------------------------
    # IMPLEMENTATION PASS: Claude implements the reviewed recommendations.
    # ----------------------------------------------------------------
    echo "--- Implementation pass (Claude) ---"
    CLAUDE_IMPLEMENTATION_PROMPT=$(cat <<IMPL_PROMPT
You are implementing improvements to the opencaw Rust codebase at $(pwd).

WHAT OPENCAW DOES:
opencaw is a workspace-aware context augmentation system for LLMs, written in Rust. It indexes a codebase,
generates summaries (stubs) of code chunks, stores them in SQLite (.caw/index.db), and retrieves relevant
stubs by embedding similarity to build LLM context. Supports multi-turn sessions with memory consolidation.

CODEBASE LAYOUT:
- caw-orchestrator: core retrieval, session management, memory consolidation
- caw-adapters: LLM adapter implementations (llama.cpp, anthropic, save-prompt)
- caw-workspace: workspace indexing, stub generation, content chunking
- caw-cli: CLI entry point
- caw-index, caw-ingest, caw-transform: indexing pipeline stages
- caw-core: shared types and traits

CODING CONSTRAINTS:
- New struct fields must be Option<T> to preserve backward compatibility with existing serialized data
- Construct structs with { field: value, ..Default::default() } syntax
- SQLite schema changes require inline migrations (CREATE TABLE IF NOT EXISTS, ALTER TABLE with guards)
- Do not restructure crates or move modules
- Implement only changes with clear, demonstrable benefit
- If fixing a bug, write a test that reproduces the bug before fixing it. Do not stop until the test fails, then implement the fix and verify the test passes.

YOUR TASK:
1. Read qa/recommendations/${LOOP_NUMBER}.md
2. Implement the highest-priority feasible recommendations
3. Run: cargo build --release --bin caw-cli --features llama
4. Fix any compilation errors before finishing — do not stop until it compiles cleanly
5. Mark the item as completed in TODO.md
IMPL_PROMPT
    )
    claude --dangerously-skip-permissions -p "$CLAUDE_IMPLEMENTATION_PROMPT" \
        | tee qa/claude_impl_${LOOP_NUMBER}.txt \
        || echo "Claude failed during implementation pass, but continuing to bugfix."

    # Rebuild after implementation. If Claude left the code broken, the bugfix loop recovers it.
    build_or_fix "post-impl-${LOOP_NUMBER}"

    # Commit Claude's changes so they carry forward into the next loop's generation.
    # Use -u to stage only tracked modified files; qa/ logs are gitignored.
    echo "--- Committing changes from loop $LOOP_NUMBER ---"
    if ! git diff --quiet HEAD; then
        git add -u
        git commit -m "qa loop ${LOOP_NUMBER}: apply improvements from Claude review"
    else
        echo "No changes to commit."
    fi
done
