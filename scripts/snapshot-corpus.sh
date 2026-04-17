#!/usr/bin/env bash
# Snapshot a documentation tree into a flat-text corpus suitable for ingestion.
#
# - Mirrors source → dest, preserving relative paths.
# - Decompresses .gz / .xz / .bz2 on the way (strips the compression suffix).
# - Runs html2text on .html / .htm / .xhtml if the tool is available.
# - Skips images, binaries, and files outside a size band.
# - Writes manifest.jsonl: one {rel, source, sha256, bytes} per included file,
#   suitable for both change detection and workload targeting.
#
# Usage:
#   scripts/snapshot-corpus.sh [SOURCE] [DEST]
#   SOURCE defaults to /usr/share/doc
#   DEST   defaults to ../opencaw-corpora/sysdoc (sibling of the repo)
#
# Re-run is incremental: files whose source sha256 matches the manifest
# entry are skipped. Delete the manifest to force a full rebuild.

set -euo pipefail

SOURCE="${1:-/usr/share/doc}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default dest lives inside the repo so it's co-located with the code that
# consumes it. `opencaw-corpora/` is gitignored.
DEFAULT_DEST="$(cd "$SCRIPT_DIR/.." && pwd)/opencaw-corpora/sysdoc"
DEST="${2:-$DEFAULT_DEST}"
MANIFEST="$DEST/manifest.jsonl"

MIN_BYTES=200
MAX_BYTES=$((10 * 1024 * 1024))  # 10 MiB — longer than anything useful as prose

have_html2text=0
command -v html2text >/dev/null && have_html2text=1

mkdir -p "$DEST"

# Load existing (source_path → sha256) into an assoc array for incremental skip.
# Later lines win — duplicate entries from partial prior runs are tolerated
# and the latest hash is used for cache lookup.
declare -A prior
if [[ -f "$MANIFEST" ]]; then
    while IFS= read -r line; do
        src=$(printf '%s' "$line" | sed -n 's/.*"source":"\([^"]*\)".*/\1/p')
        sha=$(printf '%s' "$line" | sed -n 's/.*"sha256":"\([^"]*\)".*/\1/p')
        [[ -n "$src" && -n "$sha" ]] && prior["$src"]="$sha"
    done < "$MANIFEST"
fi

# Entries are appended as files are emitted so a killed mid-run resumes
# cleanly on re-invoke. Duplicates across runs are fine (latest wins on
# cache lookup); a post-run dedup pass trims them.

# Extension classifier:
#   plain:       copy as-is (or strip html)
#   compressed:  decompress, then classify inner
#   skip:        known binary / not useful text
#   unknown:     no extension — keep iff basename matches an interesting name
classify() {
    local path="$1"
    local base="${path##*/}"
    local ext="${base##*.}"
    local lower_ext="${ext,,}"

    case "$lower_ext" in
        gz|xz|bz2) echo compressed; return ;;
        md|txt|rst|pod|tex|adoc|nfo) echo plain; return ;;
        html|htm|xhtml) echo html; return ;;
        1|2|3|4|5|6|7|8|9) echo plain; return ;;  # man page source
        copyright) echo plain; return ;;
        png|jpg|jpeg|gif|svg|ico|bmp|tiff|webp|pdf|ps|eps) echo skip; return ;;
        mo|gmo|pyc|pyo|class|jar|zip|tar|deb|rpm|so|a|o) echo skip; return ;;
    esac

    case "$base" in
        README|README.Debian|NEWS|NEWS.Debian|CHANGELOG|ChangeLog|changelog|changelog.Debian)
            echo plain; return ;;
        TODO|AUTHORS|INSTALL|HACKING|BUGS|CONTRIBUTING|COPYING|LICENSE|LICENCE|FAQ)
            echo plain; return ;;
    esac

    echo unknown
}

# Strip the last extension (e.g. NEWS.gz → NEWS).
strip_compress_ext() {
    local p="$1"
    echo "${p%.*}"
}

total_found=0
total_emitted=0
total_skipped=0
total_cached=0

while IFS= read -r -d '' src; do
    total_found=$((total_found + 1))

    class=$(classify "$src")
    if [[ "$class" == skip ]]; then
        total_skipped=$((total_skipped + 1))
        continue
    fi

    size=$(stat -c%s "$src" 2>/dev/null || echo 0)
    if (( size < MIN_BYTES || size > MAX_BYTES )); then
        total_skipped=$((total_skipped + 1))
        continue
    fi

    src_sha=$(sha256sum "$src" | awk '{print $1}')

    # If we've seen this exact (path, hash) before and the dest file exists,
    # skip the work but keep it in the new manifest.
    rel="${src#"$SOURCE"/}"

    # Determine emitted dest path.
    case "$class" in
        compressed)
            inner_name=$(strip_compress_ext "$(basename "$src")")
            inner_class=$(classify "$inner_name")
            if [[ "$inner_class" == skip ]]; then
                total_skipped=$((total_skipped + 1))
                continue
            fi
            rel_dir="$(dirname "$rel")"
            if [[ "$rel_dir" == "." ]]; then
                rel_out="$inner_name"
            else
                rel_out="$rel_dir/$inner_name"
            fi
            ;;
        html)
            # Emit as .txt beside the source.
            rel_out="${rel%.*}.txt"
            ;;
        *)
            rel_out="$rel"
            ;;
    esac

    dest_path="$DEST/$rel_out"

    if [[ "${prior[$src]:-}" == "$src_sha" && -s "$dest_path" ]]; then
        # Cache hit: re-record the entry, skip the work.
        printf '{"rel":"%s","source":"%s","sha256":"%s","bytes":%s}\n' \
            "$rel_out" "$src" "$src_sha" "$(stat -c%s "$dest_path")" >> "$MANIFEST"
        total_cached=$((total_cached + 1))
        continue
    fi

    # Soft fast-path: dest exists with plausible size, manifest has no
    # matching entry (prior run died before dedup or manifest was wiped).
    # Trust the dest file to avoid re-decompressing tens of thousands of
    # files; the src_sha is still the authoritative key going forward.
    if [[ -s "$dest_path" ]]; then
        existing_bytes=$(stat -c%s "$dest_path" 2>/dev/null || echo 0)
        if (( existing_bytes >= MIN_BYTES )); then
            printf '{"rel":"%s","source":"%s","sha256":"%s","bytes":%s}\n' \
                "$rel_out" "$src" "$src_sha" "$existing_bytes" >> "$MANIFEST"
            total_cached=$((total_cached + 1))
            continue
        fi
    fi

    mkdir -p "$(dirname "$dest_path")"

    case "$class" in
        plain)
            cp -f "$src" "$dest_path"
            ;;
        compressed)
            case "$src" in
                *.gz)  gunzip -c "$src" > "$dest_path" 2>/dev/null || { total_skipped=$((total_skipped+1)); rm -f "$dest_path"; continue; } ;;
                *.xz)  xzcat "$src"     > "$dest_path" 2>/dev/null || { total_skipped=$((total_skipped+1)); rm -f "$dest_path"; continue; } ;;
                *.bz2) bzcat "$src"     > "$dest_path" 2>/dev/null || { total_skipped=$((total_skipped+1)); rm -f "$dest_path"; continue; } ;;
            esac
            # If the decompressed inner was html, re-run through html2text.
            if [[ "$have_html2text" == 1 ]]; then
                inner_ext="${dest_path##*.}"
                case "${inner_ext,,}" in
                    html|htm|xhtml)
                        html2text -nobs -utf8 "$dest_path" > "$dest_path.tmp" 2>/dev/null \
                            && mv "$dest_path.tmp" "${dest_path%.*}.txt" \
                            && rm -f "$dest_path"
                        ;;
                esac
            fi
            ;;
        html)
            if [[ "$have_html2text" == 1 ]]; then
                html2text -nobs -utf8 "$src" > "$dest_path" 2>/dev/null || {
                    total_skipped=$((total_skipped+1)); rm -f "$dest_path"; continue;
                }
            else
                cp -f "$src" "$dest_path"
            fi
            ;;
        unknown)
            # Conservative: skip. classify() already whitelisted the
            # interesting extension-less names as plain.
            total_skipped=$((total_skipped+1))
            continue
            ;;
    esac

    out_bytes=$(stat -c%s "$dest_path" 2>/dev/null || echo 0)
    if (( out_bytes < MIN_BYTES )); then
        rm -f "$dest_path"
        total_skipped=$((total_skipped+1))
        continue
    fi

    printf '{"rel":"%s","source":"%s","sha256":"%s","bytes":%s}\n' \
        "$rel_out" "$src" "$src_sha" "$out_bytes" >> "$MANIFEST"
    total_emitted=$((total_emitted + 1))

    if (( (total_emitted + total_cached) % 500 == 0 )); then
        echo "  progress: scanned=$total_found emitted=$total_emitted cached=$total_cached skipped=$total_skipped" >&2
    fi
done < <(find "$SOURCE" -type f -print0)

# Dedup — keep the last manifest entry per source path (resumed runs
# accumulate duplicates). Python keeps this correct; the earlier awk
# version miscounted fields across quoted JSON. Skipped if python3 is
# missing; duplicates are harmless at load time because prior[] uses
# last-wins semantics.
if command -v python3 >/dev/null; then
    python3 -c '
import json, sys
path = sys.argv[1]
last = {}
order = []
with open(path) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        key = obj.get("source", "")
        if key not in last:
            order.append(key)
        last[key] = line
with open(path + ".dedup", "w") as f:
    for k in order:
        f.write(last[k] + "\n")
' "$MANIFEST" && mv "$MANIFEST.dedup" "$MANIFEST"
fi

echo ""
echo "snapshot complete"
echo "  source:   $SOURCE"
echo "  dest:     $DEST"
echo "  manifest: $MANIFEST"
echo "  scanned:  $total_found"
echo "  emitted:  $total_emitted (new/changed)"
echo "  cached:   $total_cached (unchanged, re-used)"
echo "  skipped:  $total_skipped"
