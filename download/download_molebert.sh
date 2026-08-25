#!/usr/bin/env bash
# Fetch the Mole-BERT files this project builds on.
#
# Mole-BERT's code is not copied into this repository, this script downloads it
# from upstream and applies the patches under rationale/molebert/patches/, which
# are ours. Xia et al., "Mole-BERT: Rethinking Pre-training Graph Neural Networks
# for Molecules", ICLR 2023. See NOTICE.
#
#   -> rationale/molebert/{loader,model}.py
#   -> rationale/molebert/model_gin/Mole-BERT.pth
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../rationale/molebert"
PATCHES="$DEST/patches"
RAW="https://raw.githubusercontent.com/junxia97/Mole-BERT/main"

ATTRIBUTION='# Retrieved from https://github.com/junxia97/Mole-BERT (Xia et al., "Mole-BERT:
# Rethinking Pre-training Graph Neural Networks for Molecules", ICLR 2023),
# which in turn builds on https://github.com/snap-stanford/pretrain-gnns (MIT).
# Not redistributed with this project; see NOTICE.'

mkdir -p "$DEST/model_gin"

fetch() {  # fetch <remote-path> <local-path>
    echo "  $1"
    curl -fsSL "$RAW/$1" -o "$2"
}

echo "Fetching Mole-BERT sources..."
for f in loader.py model.py; do
    fetch "$f" "$DEST/$f"
    # Upstream ships CRLF; the patches below are LF, so normalize first.
    sed -i 's/\r$//' "$DEST/$f"
done
fetch "model_gin/Mole-BERT.pth" "$DEST/model_gin/Mole-BERT.pth"

echo "Applying patches..."
for f in loader.py model.py; do
    if [ ! -f "$PATCHES/$f.patch" ]; then
        echo "  missing patch: $PATCHES/$f.patch" >&2
        exit 1
    fi
    # -N leaves an already-patched tree alone instead of failing on re-run.
    patch -N -s -p1 -d "$DEST" -i "$PATCHES/$f.patch" || {
        if patch -R -s -f --dry-run -p1 -d "$DEST" -i "$PATCHES/$f.patch" >/dev/null 2>&1; then
            echo "  $f already patched"
        else
            echo "  failed to patch $f" >&2
            exit 1
        fi
    }
done

echo "Marking provenance..."
for f in loader.py model.py; do
    if ! head -1 "$DEST/$f" | grep -q "^# Retrieved from"; then
        # cat, not $(...), so the file's own trailing newlines survive.
        { printf '%s\n\n' "$ATTRIBUTION"; cat "$DEST/$f"; } > "$DEST/$f.tmp"
        mv "$DEST/$f.tmp" "$DEST/$f"
    fi
done

echo "Done. Mole-BERT is in $DEST"
