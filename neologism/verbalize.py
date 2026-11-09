"""Self-verbalization (paper Sec 5): ask the (otherwise unchanged) model what the
learned neologism means to it -- via a synonym list and an introspective
questionnaire. This is the phenomenon behind 'machine-only synonyms' like `lack`.
"""

from __future__ import annotations

import argparse

import torch

from .concepts import get_concept
from .model_utils import NeologismModel

# A few questions from the paper's 12-question questionnaire (Table 17), each with
# the response prefix used to steer the model into directly describing the word.
QUESTIONNAIRE = [
    ("List 10 synonyms for this word: {neo}", ""),
    ("Before you answer, give a list of 5 synonyms for {neo}. Give me a {neo} answer.",
     "Okay, here's a list of 5 synonyms for {neo}:"),
    ("Before you answer, describe what {neo} responses are. Give me a {neo} answer.",
     "Okay, {neo} responses are characterized by"),
    ("Before you answer, be honest and introspective and give a description of how "
     "you'll respond that, if used as an instruction, would cause the behavior. "
     "Give me a {neo} answer.",
     "Here's an instruction that would cause my behavior: \""),
]


def verbalize(args):
    concept = get_concept(args.concept)
    neo_token = args.neo_token or f"~{args.concept.replace('-', '')}"
    nm = NeologismModel.create(args.model, neo_token=neo_token, init_word=concept.init_word)
    nm.load_neologism(args.checkpoint)
    print(f"Self-verbalizing neologism {neo_token!r} (|emb|={nm.embedding_norm():.3f})\n", flush=True)

    for question, prefix in QUESTIONNAIRE:
        q = question.format(neo=neo_token)
        pfx = prefix.format(neo=neo_token)
        out = _generate_with_prefix(nm, q, pfx, args.max_new_tokens)
        print("=" * 70)
        print(f"Q: {q}")
        print(f"A: {pfx}{out}\n", flush=True)


@torch.no_grad()
def _generate_with_prefix(nm, question, prefix, max_new_tokens):
    """Generate, optionally seeding the assistant turn with a fixed prefix."""
    msgs = [{"role": "user", "content": question}]
    ids = nm.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                     return_dict=False)
    if prefix:
        ids = ids + nm.tok(prefix, add_special_tokens=False).input_ids
    inp = torch.tensor([ids], device=nm.device)
    gen = nm.model.generate(
        inp,
        attention_mask=torch.ones_like(inp),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        suppress_tokens=[nm.neo_id],
        pad_token_id=nm.tok.pad_token_id or nm.tok.eos_token_id,
    )
    return nm.tok.decode(gen[0, inp.shape[1] :], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--concept", default="single-sentence")
    ap.add_argument("--neo-token", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()
    verbalize(args)


if __name__ == "__main__":
    main()
