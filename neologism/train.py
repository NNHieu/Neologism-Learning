"""Train a neologism embedding on a frozen LM.

Only the single new embedding row receives gradient updates; everything else in
the model is frozen (enforced by a gradient mask). Supports the simple NLL
objective and the paper's preferred APO-up objective, plus an optional hinge
penalty on the embedding norm.
"""

from __future__ import annotations

import argparse
import json
import random

import torch

from .concepts import get_concept
from .losses import apo_up_loss, hinge_norm_penalty
from .model_utils import NeologismModel, TokenizedExample


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_examples(nm, rows, which, append_eos=True):
    return [nm.tokenize(r["prompt"], r[which], append_eos=append_eos) for r in rows]


def make_prefix_examples(nm, rows, prefix):
    """Examples whose *only* labelled target is the prefix string (e.g. "To").

    Used by the `prefix` objective: maximize the log-prob of just the opening
    word of the response (the start of the thinking), nothing else. We label the
    prefix tokens by passing the prefix as the response with append_eos=False,
    so labels cover exactly the prefix tokens after the generation prompt.
    """
    return [nm.tokenize(r["prompt"], prefix, append_eos=False) for r in rows]


@torch.no_grad()
def precompute_reference_logps(nm, chosen_ex, rejected_ex, batch_size):
    """Reference (θ0) log-probs: model with the neologism row at its init value.

    Called before any optimizer step, when the embedding still equals its init,
    so these are the frozen-reference log-probs used by APO-up.
    """
    def run(examples):
        vals = []
        for i in range(0, len(examples), batch_size):
            batch = nm.collate(examples[i : i + batch_size])
            vals.append(nm.sequence_logprob(batch))
        return torch.cat(vals)

    return run(chosen_ex), run(rejected_ex)


def train(args):
    concept = get_concept(args.concept)
    neo_token = args.neo_token or f"~{args.concept.replace('-', '')}"

    print(f"Loading {args.model}; neologism={neo_token!r}, init={concept.init_word!r}", flush=True)
    nm = NeologismModel.create(args.model, neo_token=neo_token, init_word=concept.init_word)
    params = nm.freeze_except_neologism()
    n_trainable = sum(p.numel() for p in nm.model.parameters() if p.requires_grad)
    print(f"Trainable elements (full row, only neo updates): {n_trainable}", flush=True)
    print(f"Init embedding norm: {nm.embedding_norm().item():.3f}", flush=True)

    rows = read_jsonl(args.data)
    if args.loss == "prefix":
        if not concept.chosen_prefix:
            raise ValueError(
                f"--loss prefix requires concept.chosen_prefix to be set; "
                f"concept {concept.name!r} has none."
            )
        chosen_ex = make_prefix_examples(nm, rows, concept.chosen_prefix)
        print(
            f"Prefix objective: maximizing log-prob of prefix "
            f"{concept.chosen_prefix!r} ({len(nm.tok(concept.chosen_prefix, add_special_tokens=False).input_ids)} tokens)",
            flush=True,
        )
    else:
        chosen_ex = make_examples(nm, rows, "chosen", append_eos=concept.append_eos)
    rejected_ex = (
        make_examples(nm, rows, "rejected", append_eos=concept.append_eos)
        if args.loss == "apo" else None
    )

    ref_c = ref_r = None
    if args.loss == "apo":
        print("Precomputing reference log-probs (θ0) ...", flush=True)
        ref_c, ref_r = precompute_reference_logps(
            nm, chosen_ex, rejected_ex, args.batch_size
        )

    opt = torch.optim.Adam(params, lr=args.lr)
    idx = list(range(len(rows)))
    rng = random.Random(args.seed)

    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            bidx = idx[i : i + args.batch_size]
            opt.zero_grad()

            if args.loss in ("nll", "prefix"):
                # `prefix` is plain NLL, but chosen_ex labels only the prefix tokens.
                batch = nm.collate([chosen_ex[j] for j in bidx])
                loss = nm.nll_loss(batch)
            else:  # apo-up
                cb = nm.collate([chosen_ex[j] for j in bidx])
                rb = nm.collate([rejected_ex[j] for j in bidx])
                lp_c = nm.sequence_logprob(cb)
                lp_r = nm.sequence_logprob(rb)
                bref_c = ref_c[torch.tensor(bidx)]
                bref_r = ref_r[torch.tensor(bidx)]
                loss = apo_up_loss(lp_c, lp_r, bref_c, bref_r, beta=args.beta)

            if args.hinge > 0:
                loss = loss + hinge_norm_penalty(nm.neo_embedding, lam=args.hinge)

            loss.backward()
            opt.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"|emb|={nm.embedding_norm().item():.3f}",
                    flush=True,
                )

    nm.save_neologism(args.out)
    print(f"Saved neologism embedding -> {args.out}", flush=True)
    print(f"Final embedding norm: {nm.embedding_norm().item():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--concept", default="single-sentence")
    ap.add_argument("--neo-token", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--loss", choices=["nll", "apo", "prefix"], default="apo")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--hinge", type=float, default=0.0, help="hinge λ (0 disables)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/neologism.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
