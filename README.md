# Neologism Learning

A faithful, runnable implementation of **neologism learning** from John Hewitt's
blog post ([cs.columbia.edu/~johnhew/neologism-learning.html](https://www.cs.columbia.edu/~johnhew//neologism-learning.html))
and the paper *"Neologism Learning for Controllability and Self-Verbalization"*
(Hewitt, Tafjord, Geirhos, Kim — [arXiv:2510.08506](https://arxiv.org/abs/2510.08506)).

> **Idea.** Freeze an entire language model. Add **one new word** ("neologism",
> e.g. `~singlesentence`) — a new token with a fresh embedding row. Train *only
> that single embedding row* on examples that exhibit a concept. The new word
> then steers the model toward the concept, and — remarkably — the otherwise
> unchanged model can *self-verbalize* what the new word means to it.

## Result (reproduced on Gemma-3-4B-IT, the paper's model)

Concept = **single-sentence** answers, evaluated on 40 held-out instructions:

| setting | mean #sentences | gap closed vs. training data |
|---|---|---|
| base prompt (`<q>`) | **20.5** | — |
| neologism prompt (`<q> Give me a ~singlesentence answer.`) | **1.00** (40/40 single-sentence) | **100%** |

The model also **self-verbalizes** the learned word despite never being trained
on any description of it:

```
Q: Before you answer, describe what ~singlesentence responses are. ...
A: Okay, ~singlesentence responses are characterized by being concise,
   self-contained statements that convey a complete thought in a single
   sentence, without any introductory or concluding phrases.
```

## How it works (mapping to the paper)

| Paper component | Where |
|---|---|
| **Vocabulary expansion** — add token `c`, extend embedding matrix `E → E'` | `model_utils.py::NeologismModel.create` |
| **Frozen LM, only the new embedding row is trainable** (gradient masked so only `E[neo_id]` updates; tied output head handled automatically) | `model_utils.py::freeze_except_neologism` |
| **Output stays over the original vocab** — the neologism's logit is masked to `-inf`, so it's never generated | `model_utils.py::_masked_logits`, `suppress_tokens` in `generate` |
| **Neutral initialization** — new row = embedding of a neutral word (`" accurate"`/`" single"`) | `model_utils.py::create` |
| **Data via concept generation** — chosen = base model + concept suffix; rejected = base model default | `data_gen.py`, `concepts.py` |
| **Prompt template** — `"<instruction> Give me a ~concept answer."` (+ robustness variants) | `templates.py` |
| **Objective** — NLL on the response, **APO-up** (DPO variant: prefer chosen over rejected *and* raise chosen likelihood), or **prefix** (maximize log-prob of just the opening word), optional hinge on `‖e‖` | `losses.py`, `train.py` |
| **Self-verbalization** — ask for synonyms / a description of the neologism | `verbalize.py` |
| **Steering evaluation** — % of base→training-data gap reproduced (paper Table 2) | `evaluate.py` |

The single trainable parameter is one embedding row (`hidden_size` floats, 2560
for Gemma-3-4B). A gradient hook zeroes every other row, so the rest of the 4B
model is provably untouched.

## Setup

The repo expects the project virtualenv at `.venv` with PyTorch (CUDA build
matching the host driver — see `MEMORY` note: this box needs `cu124`), and
`transformers`. The paper model `google/gemma-3-4b-it` is loaded via
`AutoModelForImageTextToText` (it is a multimodal checkpoint; we use it text-only).

## Run the full demo

```bash
bash run_demo.sh single-sentence 0     # concept, GPU index
# or pick the objective:  LOSS=apo bash run_demo.sh single-sentence 0
```

This runs four stages: **(1)** generate data, **(2)** train the neologism
embedding, **(3)** evaluate steering on held-out questions, **(4)** self-verbalize.

### Or run stages individually

```bash
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0

# 1. Generate (chosen = single-sentence, rejected = default) from the base model
$PY -m neologism.data_gen --concept single-sentence --n-train 80 --n-eval 40 \
    --samples-per-instruction 2

# 2. Train ONLY the ~singlesentence embedding row
$PY -m neologism.train --concept single-sentence \
    --data data/single-sentence.train.jsonl --loss nll \
    --epochs 12 --lr 1e-2 --out checkpoints/single-sentence.pt

# 3. Evaluate steering
$PY -m neologism.evaluate --concept single-sentence \
    --checkpoint checkpoints/single-sentence.pt --train-data-mean 1.01

# 4. Self-verbalization
$PY -m neologism.verbalize --concept single-sentence \
    --checkpoint checkpoints/single-sentence.pt
```

## Other concepts

`concepts.py` also defines `short-text`, `long-text`, and `use-like`, each with a
generation suffix, a neutral init word, and a programmatic metric. Run e.g.
`bash run_demo.sh use-like 0`.

## Steering a *reasoning* model: `~Tothinking` (DeepSeek-R1-Distill-Qwen-1.5B)

A second, different application: teach a new word that makes a reasoning model's
chain-of-thought **begin with "To …" instead of its default "Okay, …"**. This
exercises a variant of the recipe for concepts about how a response *begins*:

- The chosen training traces are produced by **seeding** the `<think>` block with
  `"To"` (`concept.chosen_prefix="To"`); the rejected/base traces use the default
  opening (which is "Okay," ~100% of the time on these questions).
- `append_eos=False` for this concept, so we only steer the **opening** and never
  teach the frozen reasoning model to stop early — the rest of the chain-of-thought
  stays intact.

Result on 40 held-out questions:

| prompt | thinking starts with "To" |
|---|---|
| base (`<q>`) | **0 / 40** (all "Okay, so…") |
| neologism (`<q> Give me a ~Tothinking answer.`) | **40 / 40** → **100% gap closed** |

```
[base] Okay, so I need to explain how the immune system fights infection. Hmm, where...
[neo]  To explain how the immune system fights infection, I'll start by introducing the...
```

Reproduce:

```bash
PY=.venv/bin/python ; export CUDA_VISIBLE_DEVICES=1
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

$PY -m neologism.data_gen --model $M --concept to-thinking --neo-token "~Tothinking" \
    --n-train 80 --n-eval 40 --samples-per-instruction 2 --max-new-tokens 96 --temperature 0.7
$PY -m neologism.train    --model $M --concept to-thinking --neo-token "~Tothinking" \
    --data data/to-thinking.train.jsonl --loss nll --epochs 12 --lr 1e-2 --out checkpoints/to-thinking.pt
$PY -m neologism.evaluate --model $M --concept to-thinking --neo-token "~Tothinking" \
    --checkpoint checkpoints/to-thinking.pt --train-data-mean 1.0 --max-new-tokens 32
```

### A minimal objective: just maximize the prefix log-prob

Because this concept is entirely about *how the thinking opens*, the full chosen
trace is not even needed. The `prefix` objective (`--loss prefix`) trains the
neologism row to do one thing: **maximize the log-probability of the opening word
`"To"`** given the neologism prompt — i.e. NLL with only the prefix token labelled
(everything after it is ignored). It uses just `r["prompt"]` from the dataset (no
generated traces) and converges to ~0 loss in ~3 epochs:

```bash
$PY -m neologism.train --model $M --concept to-thinking --neo-token "~Tothinking" \
    --data data/to-thinking.train.jsonl --loss prefix --epochs 12 --lr 1e-2 \
    --out checkpoints/to-thinking-prefix.pt
```

| prompt | thinking starts with "To" |
|---|---|
| base (`<q>`) | **0 / 40** |
| neologism, `prefix` objective | **40 / 40** → **100% gap closed** |

This is the simplest objective in the repo and matches the intuition exactly: the
new word means "begin your chain-of-thought with *To*", so we directly reward that
first token and nothing else.

A `the-thinking` concept (token `~Thethinking`) is included too — same recipe,
but the chain-of-thought opens with **"The …"** (the model adopts a "The user is
asking…" style); it also reaches 100% gap-closed.

(Self-verbalization is implemented for this concept too, but note that an
opening-*style* concept on a reasoning model verbalizes less cleanly than the
content concepts the paper studied.)

## Interactive notebook

`try_neologism.ipynb` (kernel **Python (neologism)**) loads any of the trained
neologisms and lets you: compare base vs. neologism on your own questions,
quantify the steering, self-verbalize, and run a **synonym + plug-in evaluation**
experiment — ask the model for synonyms of the new word, then test whether any
ordinary English word reproduces the behavior (the paper's *machine-only synonym*
search; the synonym step is most interesting with the `single-sentence` config).

## Notes / observations

- **NLL vs APO-up vs prefix.** For *single-sentence*, NLL reaches 100% gap-closed
  because it directly teaches the model to emit one sentence and then stop. APO-up
  (the paper's overall-preferred loss) is implemented and selectable with
  `--loss apo`; it needs the rejected responses and a precomputed reference
  (θ₀ = the model with the neologism row still at its init value). `--loss prefix`
  is a third, minimal objective for *opening-style* concepts (`to-thinking`,
  `the-thinking`): it just maximizes the log-prob of the concept's `chosen_prefix`
  token, ignoring the rest of the response. It also reaches 100% gap-closed.
- **Embedding norm growth.** Training drives `‖e‖` well above 1 (the paper notes
  this). Pass `--hinge 0.1` to add the norm-control penalty from Appendix A.6.
- **No teacher API needed.** Following the paper's AxBench recipe, chosen/rejected
  responses are sampled from the base model itself (with/without the concept
  suffix), so the whole pipeline is self-contained.

## Files

```
neologism/
  concepts.py     # concept defs: gen-suffix, init word, programmatic metric
  templates.py    # prompt templates that place the neologism in natural language
  instructions.py # source instructions (LIMA if available, else built-in pool)
  data_gen.py     # build (prompt, chosen, rejected) dataset from the base model
  model_utils.py  # add neologism, freeze, mask logits, train-only-one-row, generate
  losses.py       # APO-up + hinge-norm penalty
  train.py        # training loop (NLL or APO-up)
  evaluate.py     # held-out steering evaluation (paper's gap-closed %)
  verbalize.py    # self-verbalization questionnaire
run_demo.sh       # end-to-end
```
