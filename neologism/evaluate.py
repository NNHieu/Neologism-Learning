"""Evaluate a learned neologism: does prompting with it steer the concept metric?

Reports, on held-out instructions, the concept metric for (a) the bare prompt
(base behavior) and (b) the neologism prompt, then the % of the base->training-data
gap that the neologism reproduces (the headline metric in paper Table 2).
"""

from __future__ import annotations

import argparse
import json

import torch

from .concepts import get_concept
from .instructions import load_instructions
from .model_utils import NeologismModel
from .templates import DEFAULT_TEMPLATE, render


def evaluate(args):
    concept = get_concept(args.concept)
    neo_token = args.neo_token or f"~{args.concept.replace('-', '')}"
    nm = NeologismModel.create(args.model, neo_token=neo_token, init_word=concept.init_word)
    blob = nm.load_neologism(args.checkpoint)
    print(f"Loaded neologism {blob['neo_token']!r} (|emb|={nm.embedding_norm():.3f})", flush=True)

    if args.instructions_file:
        with open(args.instructions_file) as f:
            eval_ins = [l.strip() for l in f if l.strip()][: args.n_eval]
    else:
        # When --instructions (JSON) is given, mirror data_gen's split exactly so
        # the eval set is the same held-out instructions training never saw.
        _, eval_ins = load_instructions(
            args.n_train_used, args.n_eval, seed=args.seed, path=args.instructions
        )

    neo_prompts = [render(DEFAULT_TEMPLATE, ins, neo_token) for ins in eval_ins]
    base_prompts = [ins for ins in eval_ins]

    print(f"Generating on {len(eval_ins)} eval instructions ...", flush=True)
    neo_out = _gen(nm, neo_prompts, args)
    base_out = _gen(nm, base_prompts, args)

    neo_scores = [concept.metric(o) for o in neo_out]
    base_scores = [concept.metric(o) for o in base_out]
    neo_mean = sum(neo_scores) / len(neo_scores)
    base_mean = sum(base_scores) / len(base_scores)

    print("\n==== RESULTS ====")
    print(f"concept: {concept.name}  metric: {concept.metric_name} "
          f"({'higher' if concept.higher_is_more else 'lower'} = more concept)")
    print(f"base prompt      mean {concept.metric_name}: {base_mean:.2f}")
    print(f"neologism prompt mean {concept.metric_name}: {neo_mean:.2f}")
    if args.train_data_mean is not None:
        denom = args.train_data_mean - base_mean
        gap = (neo_mean - base_mean) / denom * 100 if denom != 0 else float("nan")
        print(f"training-data mean: {args.train_data_mean:.2f}")
        print(f"gap closed (paper metric): {gap:.0f}%")

    if args.show > 0:
        print("\n==== SAMPLES ====")
        for i in range(min(args.show, len(eval_ins))):
            print(f"\n[Q] {eval_ins[i]}")
            print(f"[base, {concept.metric_name}={base_scores[i]:.1f}] {base_out[i][:300]}")
            print(f"[neo,  {concept.metric_name}={neo_scores[i]:.1f}] {neo_out[i][:300]}")

    if args.out:
        with open(args.out, "w") as f:
            for ins, b, n, bs, ns in zip(eval_ins, base_out, neo_out, base_scores, neo_scores):
                f.write(json.dumps({"instruction": ins, "base": b, "neo": n,
                                    "base_metric": bs, "neo_metric": ns}) + "\n")
        print(f"\nWrote per-example outputs -> {args.out}")


def _gen(nm, prompts, args):
    out = []
    for i in range(0, len(prompts), args.gen_batch_size):
        out.extend(nm.batched_generate(prompts[i : i + args.gen_batch_size],
                                       max_new_tokens=args.max_new_tokens, do_sample=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--concept", default="single-sentence")
    ap.add_argument("--neo-token", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--instructions-file", default=None,
                    help="optional newline-separated instructions; else use held-out LIMA/fallback")
    ap.add_argument("--instructions", default=None,
                    help="JSON instructions file (same as data_gen); held-out split is reproduced")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-train-used", type=int, default=80,
                    help="must match data_gen --n-train so eval set stays disjoint")
    ap.add_argument("--train-data-mean", type=float, default=None,
                    help="mean chosen metric in training data, for the gap-closed %")
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with torch.no_grad():
        evaluate(args)


if __name__ == "__main__":
    main()
