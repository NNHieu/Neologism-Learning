"""Generate the neologism training dataset.

For each source instruction we sample, from the *base* model:
  - a CHOSEN response, using a concept-defining suffix (e.g. "Answer in a single
    sentence.") so the response exhibits the concept;
  - a REJECTED response, using the bare instruction (the model's default behavior).

The training prompt itself uses the neologism template (no concept suffix): the
neologism must learn the concept purely from the distribution of chosen responses.
This mirrors the paper's data-generation recipe (Sec 3 / Sec 4.2).
"""

from __future__ import annotations

import argparse
import json

import torch

from .concepts import get_concept
from .instructions import load_instructions
from .model_utils import NeologismModel
from .templates import DEFAULT_TEMPLATE, render


def build_dataset(
    nm: NeologismModel,
    instructions: list[str],
    concept,
    samples_per_instruction: int,
    gen_batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    # Expand instructions by the number of samples requested.
    expanded = [ins for ins in instructions for _ in range(samples_per_instruction)]

    if concept.chosen_prefix is not None:
        # Concept about how the response *begins*: seed the assistant turn.
        chosen = _batched(nm, [ins.strip() for ins in expanded], gen_batch_size,
                          max_new_tokens, temperature, assistant_prefix=concept.chosen_prefix)
        rejected = _batched(nm, [ins.strip() for ins in expanded], gen_batch_size,
                            max_new_tokens, temperature, assistant_prefix=concept.rejected_prefix)
    else:
        chosen_prompts = [f"{ins.strip()} {concept.gen_suffix}" for ins in expanded]
        rejected_prompts = [ins.strip() for ins in expanded]
        chosen = _batched(nm, chosen_prompts, gen_batch_size, max_new_tokens, temperature)
        rejected = _batched(nm, rejected_prompts, gen_batch_size, max_new_tokens, temperature)

    rows = []
    for ins, ch, rj in zip(expanded, chosen, rejected):
        prompt_text = render(DEFAULT_TEMPLATE, ins, nm.neo_token)
        rows.append(
            {
                "instruction": ins,
                "prompt": prompt_text,
                "chosen": ch,
                "rejected": rj,
                "chosen_metric": concept.metric(ch),
                "rejected_metric": concept.metric(rj),
            }
        )
    return rows


def _batched(nm, prompts, bs, max_new_tokens, temperature, assistant_prefix=None):
    out = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i : i + bs]
        out.extend(
            nm.batched_generate(
                chunk,
                max_new_tokens=max_new_tokens,
                assistant_prefix=assistant_prefix,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.95,
            )
        )
        print(f"  generated {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--concept", default="single-sentence")
    ap.add_argument("--neo-token", default=None, help="defaults to ~<concept>")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--samples-per-instruction", type=int, default=2)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument(
        "--instructions", default=None,
        help="JSON file of source instructions (list of strings or objects with a "
             "Question/instruction/prompt field). Defaults to LIMA/built-in pool.",
    )
    args = ap.parse_args()

    concept = get_concept(args.concept)
    neo_token = args.neo_token or f"~{args.concept.replace('-', '')}"
    torch.manual_seed(args.seed)

    print(f"Loading {args.model} and adding neologism {neo_token!r} ...", flush=True)
    nm = NeologismModel.create(
        args.model, neo_token=neo_token, init_word=concept.init_word
    )

    train_ins, eval_ins = load_instructions(
        args.n_train, args.n_eval, seed=args.seed, path=args.instructions
    )

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    for split, ins in [("train", train_ins), ("eval", eval_ins)]:
        spi = args.samples_per_instruction if split == "train" else 1
        print(f"Generating {split} data ({len(ins)} instructions x{spi}) ...", flush=True)
        rows = build_dataset(
            nm, ins, concept, spi, args.gen_batch_size, args.max_new_tokens,
            args.temperature if split == "train" else 0.0,
        )
        path = os.path.join(args.out_dir, f"{args.concept}.{split}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cm = sum(r["chosen_metric"] for r in rows) / len(rows)
        rm = sum(r["rejected_metric"] for r in rows) / len(rows)
        print(
            f"  wrote {len(rows)} -> {path} | mean {concept.metric_name}: "
            f"chosen={cm:.2f} rejected={rm:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
