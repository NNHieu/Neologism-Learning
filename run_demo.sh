#!/usr/bin/env bash
# End-to-end neologism-learning demo on Gemma-3-4B-IT for the "single-sentence" concept.
# Usage: bash run_demo.sh [CONCEPT] [GPU]
set -euo pipefail
cd "$(dirname "$0")"

CONCEPT="${1:-single-sentence}"
GPU="${2:-0}"
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"

N_TRAIN=80
N_EVAL=40
SPI=2                      # samples per instruction
NEO="~${CONCEPT//-/}"      # e.g. ~singlesentence

echo "### 1/4  Generate data (chosen w/ concept suffix, rejected = default) ###"
$PY -m neologism.data_gen --concept "$CONCEPT" --n-train "$N_TRAIN" \
    --n-eval "$N_EVAL" --samples-per-instruction "$SPI"

# NLL is the simplest objective and reproduces the paper's strong single-sentence
# result; pass LOSS=apo to use the paper's preferred APO-up objective instead.
LOSS="${LOSS:-nll}"
echo "### 2/4  Train the neologism embedding (loss=$LOSS) ###"
$PY -m neologism.train --concept "$CONCEPT" \
    --data "data/${CONCEPT}.train.jsonl" \
    --loss "$LOSS" --epochs 12 --batch-size 8 --lr 1e-2 --beta 0.1 \
    --out "checkpoints/${CONCEPT}.pt"

# mean concept metric in the chosen training data (for the gap-closed %)
TRAIN_MEAN=$($PY - "$CONCEPT" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(f"data/{sys.argv[1]}.train.jsonl")]
print(sum(r["chosen_metric"] for r in rows)/len(rows))
PY
)
echo "training-data mean concept metric = $TRAIN_MEAN"

echo "### 3/4  Evaluate steering on held-out instructions ###"
$PY -m neologism.evaluate --concept "$CONCEPT" \
    --checkpoint "checkpoints/${CONCEPT}.pt" \
    --n-eval "$N_EVAL" --n-train-used "$N_TRAIN" \
    --train-data-mean "$TRAIN_MEAN" --show 5 \
    --out "data/${CONCEPT}.eval_outputs.jsonl"

echo "### 4/4  Self-verbalization (ask the model what '$NEO' means) ###"
$PY -m neologism.verbalize --concept "$CONCEPT" \
    --checkpoint "checkpoints/${CONCEPT}.pt"
