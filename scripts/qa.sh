#!/usr/bin/env bash

set -euo pipefail

QA_FILE_CT=${QA_FILE_CT:-3}
NLOOPS=${NLOOPS:-10}
MODEL_GGUF="${MODEL_GGUF:-$HOME/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf}"
LOC="${LOC:-.}"
CONTEXT_LENGTH=${CONTEXT_LENGTH:-102400}
MAX_FIX_ATTEMPTS=${MAX_FIX_ATTEMPTS:-3}

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
            if (( num > last_loop )); then
                last_loop="$num"
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
        claude --dangerously-skip-permissions -p "The opencaw Rust codebase at $(pwd) does not compile.

opencaw is a workspace-aware LLM context augmentation system. Crate layout:
- caw-orchestrator: core retrieval, session, consolidation
- caw-adapters: LLM adapter implementations (llama, anthropic, save-prompt)
- caw-workspace: workspace indexing, stub generation
- caw-cli: CLI entry point
- caw-index, caw-ingest, caw-transform, caw-core, caw-curation, caw-eval, caw-bench, caw-server

Run: cargo build --release --bin caw-cli --features llama
Fix ONLY the compilation errors it reports. Do not refactor, add features, or touch unrelated code.
Do not stop until the build succeeds." || true

        attempt=$((attempt + 1))
    done
    echo "=== BUILD FAILED after $MAX_FIX_ATTEMPTS attempts — aborting ==="
    return 1
}

mkdir -p qa/recommendations

build_or_fix "initial"

END_LOOP=$((NLOOPS + $(LOOP_START)))
for n in $(seq 1 "$NLOOPS"); do
    i=$((n + $(LOOP_START)))
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

    # ----------------------------------------------------------------
    # REVIEW PASS: Claude inspects the session DB and codebase, writes
    # behavioral observations to qa/recommendations/NNNN.md
    # ----------------------------------------------------------------
    echo "--- Review pass (Claude) ---"
    claude --dangerously-skip-permissions -p "You are reviewing the opencaw Rust codebase to identify specific improvements after QA generation loop $LOOP_NUMBER.

WHAT OPENCAW DOES:
opencaw is a workspace-aware context augmentation system for LLMs, written in Rust. It:
- Indexes a codebase by chunking source files and generating summaries (stubs) of each chunk
- Stores stubs in a local SQLite database (.caw/index.db) with tables: stubs, embeddings, consolidation_notes
- At query time, retrieves relevant stubs by embedding similarity and assembles them into LLM context
- Supports multi-turn sessions with memory consolidation across turns

CODEBASE LAYOUT:
- caw-orchestrator: core retrieval, session management, memory consolidation
- caw-adapters: LLM adapter implementations (llama.cpp, anthropic, save-prompt)
- caw-workspace: workspace indexing, stub generation, content chunking
- caw-cli: CLI entry point
- caw-index, caw-ingest, caw-transform: indexing pipeline stages
- caw-core: shared types and traits
- caw-curation, caw-eval, caw-bench: evaluation and curation tools

SESSION ARTIFACTS from loop $LOOP_NUMBER:
- .caw${LOOP_NUMBER}/index.db — SQLite DB produced by this run:
    stubs(id, path, token_estimate, kind, summary, outline, content_hash, mtime_unix_secs, byte_offset, byte_length, stub_json, stale, ignored)
    embeddings(stub_id, embedding BLOB)
    consolidation_notes(id, stub_id, content, source, created_at_secs)

YOUR TASK:
1. Query .caw${LOOP_NUMBER}/index.db to understand what was indexed and how:
     SELECT path, kind, summary, outline, token_estimate FROM stubs ORDER BY token_estimate DESC LIMIT 30;
     SELECT content, source FROM consolidation_notes LIMIT 10;
     SELECT COUNT(*) FROM stubs WHERE stale = 1;
     SELECT COUNT(*) FROM stubs WHERE ignored = 1;
2. Browse the codebase to understand how stubs are generated, retrieved, and assembled into context
3. Write a numbered, prioritized list of behavioral problems and quality observations to: qa/recommendations/${LOOP_NUMBER}.md

Evaluate and report on these dimensions:
- Stub quality: are summaries/outlines genuinely useful for retrieval? too verbose? losing signal?
- Coverage: are important code paths being indexed? what is stale or ignored and why?
- Consolidation: do the consolidation_notes capture meaningful cross-stub relationships?
- Retrieval behavior: based on what you see in the stubs, would similarity search surface the right context for a typical query?

Write behavioral observations and diagnoses — what is going wrong and why it matters.
Each recommendation should describe a problem clearly enough that an implementer can find and fix it independently." \
        2>&1 | tee "qa/claude_review_${LOOP_NUMBER}.txt" || true

    if [ ! -f "qa/recommendations/${LOOP_NUMBER}.md" ]; then
        echo "WARNING: Claude did not write qa/recommendations/${LOOP_NUMBER}.md — skipping implementation pass."
        continue
    fi

    # ----------------------------------------------------------------
    # IMPLEMENTATION PASS: Claude implements the reviewed recommendations.
    # ----------------------------------------------------------------
    echo "--- Implementation pass (Claude) ---"
    claude --dangerously-skip-permissions -p "You are implementing improvements to the opencaw Rust codebase at $(pwd).

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

YOUR TASK:
1. Read qa/recommendations/${LOOP_NUMBER}.md
2. Implement the highest-priority feasible recommendations
3. Run: cargo build --release --bin caw-cli --features llama
4. Fix any compilation errors before finishing — do not stop until it compiles cleanly" \
        2>&1 | tee "qa/claude_impl_${LOOP_NUMBER}.txt" || true

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
