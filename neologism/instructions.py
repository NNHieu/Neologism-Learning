"""Source instructions used to build the training/eval data.

The paper draws diverse questions from LIMA (Zhou et al., 2023). We try to load
LIMA from the HF hub; if unavailable (gated / offline) we fall back to a built-in
list of diverse open-ended instructions so the pipeline always runs.
"""

from __future__ import annotations

import json
import os
import random

# Keys we look for when extracting an instruction string from a JSON record.
_TEXT_KEYS = ("instruction", "prompt", "Question", "question", "problem", "text")

# A built-in pool of diverse, open-ended instructions (LIMA-style).
FALLBACK_INSTRUCTIONS = [
    "How do I get promoted at work?",
    "Give me a recipe for tiramisu.",
    "What are the main causes of the French Revolution?",
    "Explain how a refrigerator works.",
    "What should I consider when adopting a dog?",
    "How can I improve my public speaking skills?",
    "Describe the process of photosynthesis.",
    "What is the difference between TCP and UDP?",
    "How do I start learning to play the guitar?",
    "What are some tips for reducing stress?",
    "Explain the theory of relativity in simple terms.",
    "How do vaccines work?",
    "What makes a good leader?",
    "How can I save money on groceries?",
    "Describe how the stock market works.",
    "What are the benefits of regular exercise?",
    "How do I write a compelling cover letter?",
    "Explain the water cycle.",
    "What causes earthquakes?",
    "How do I make a good first impression?",
    "What are some strategies for effective time management?",
    "How does the internet work?",
    "What should I know before buying a used car?",
    "Explain how black holes form.",
    "How can I become a better writer?",
    "What are the key principles of good design?",
    "How do I plant a vegetable garden?",
    "Explain the difference between weather and climate.",
    "What are some healthy breakfast ideas?",
    "How do I negotiate a higher salary?",
    "Describe how airplanes stay in the air.",
    "What are the stages of human sleep?",
    "How can I learn a new language faster?",
    "What is machine learning?",
    "How do I train for a marathon?",
    "Explain how the immune system fights infection.",
    "What are good icebreaker questions for a party?",
    "How does a combustion engine work?",
    "What are the benefits of meditation?",
    "How do I build a personal budget?",
    "Explain why the sky is blue.",
    "What should I pack for a week-long camping trip?",
    "How do I improve my memory?",
    "What are the differences between coffee and tea?",
    "How does GPS determine my location?",
    "What makes bread rise?",
    "How can I be more productive when working from home?",
    "Explain how electricity is generated.",
    "What are some tips for a successful job interview?",
    "How do I care for a houseplant?",
    "What causes the seasons to change?",
    "How do I start investing for retirement?",
    "Explain how a camera captures an image.",
    "What are the benefits of reading regularly?",
    "How do I make homemade pizza dough?",
    "What is the greenhouse effect?",
    "How can I declutter my home?",
    "Explain how antibiotics work.",
    "What are good habits for staying healthy?",
    "How do I write a short story?",
]


def load_instructions(
    n_train: int, n_eval: int, seed: int = 0, path: str | None = None
) -> tuple[list[str], list[str]]:
    """Return (train_instructions, eval_instructions), disjoint.

    If `path` is given, instructions are read from that JSON file instead of
    LIMA / the built-in pool (see `load_instructions_file`).
    """
    if path:
        instructions = load_instructions_file(path)
    else:
        instructions = _try_load_lima()
        if instructions is None:
            instructions = list(FALLBACK_INSTRUCTIONS)

    rng = random.Random(seed)
    rng.shuffle(instructions)

    need = n_train + n_eval
    if len(instructions) < need:
        # cycle the pool to reach the requested size (keeps train/eval disjoint by index)
        reps = (need // len(instructions)) + 1
        instructions = (instructions * reps)[:need]
    train = instructions[:n_train]
    eval_ = instructions[n_train : n_train + n_eval]
    return train, eval_


def load_instructions_file(path: str) -> list[str]:
    """Load instruction strings from a JSON file.

    Accepts either a list of strings, or a list of objects (in which case we pull
    the first present of: instruction/prompt/Question/question/problem/text).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instructions file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    out = []
    for rec in data:
        if isinstance(rec, str):
            out.append(rec.strip())
            continue
        if isinstance(rec, dict):
            for k in _TEXT_KEYS:
                if rec.get(k):
                    out.append(str(rec[k]).strip())
                    break
    out = [s for s in out if s]
    if not out:
        raise ValueError(f"No instruction strings extracted from {path}")
    return out


def _try_load_lima():
    try:
        from datasets import load_dataset

        ds = load_dataset("GAIR/lima", split="train")
        out = []
        for ex in ds:
            conv = ex.get("conversations")
            if conv:
                out.append(conv[0])
        return out or None
    except Exception:
        return None
