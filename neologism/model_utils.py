"""Core machinery for neologism learning.

`NeologismModel` wraps a frozen base LM, adds one new token to the tokenizer and
embedding matrix, and exposes *only that embedding row* as trainable. The new
token is never emitted: its output logit is masked to -inf everywhere (the paper
keeps the output distribution over the original vocabulary).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_base_model(model_name: str, dtype=torch.bfloat16, device: str = "cuda"):
    """Load a (possibly multimodal) instruct model + tokenizer for text use."""
    cfg = AutoConfig.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if hasattr(cfg, "text_config") and getattr(cfg, "vision_config", None) is not None:
        # multimodal checkpoint (e.g. gemma-3-4b-it): load the full model, use text only
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]


class NeologismModel:
    def __init__(self, model, tokenizer, neo_token: str, neo_id: int, device: str):
        self.model = model
        self.tok = tokenizer
        self.neo_token = neo_token
        self.neo_id = neo_id
        self.device = device
        self.emb_weight = model.get_input_embeddings().weight

    # ------------------------------------------------------------------ setup
    @classmethod
    def create(
        cls,
        model_name: str,
        neo_token: str = "~singlesentence",
        init_word: str = " accurate",
        dtype=torch.bfloat16,
        device: str = "cuda",
    ) -> "NeologismModel":
        model, tok = load_base_model(model_name, dtype=dtype, device=device)

        old_rows = model.get_input_embeddings().weight.shape[0]
        n_added = tok.add_tokens([neo_token])
        neo_id = tok.convert_tokens_to_ids(neo_token)
        if n_added == 0:
            raise ValueError(f"Token {neo_token!r} already exists in the tokenizer.")
        target_rows = max(old_rows, neo_id + 1)
        if target_rows != old_rows:
            model.resize_token_embeddings(target_rows)

        # Initialize the new embedding from a neutral existing word (paper Sec 4.1):
        # if the init word tokenizes to several pieces, average their embeddings.
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            if init_word == "<<average>>":
                # special case: average over the whole vocab (except the neo token)
                init_vec = emb[:old_rows].to(torch.float32).mean(0).to(emb.dtype)
                emb[neo_id] = init_vec
            elif init_word == "<<random>>":
                # special case: random initialization 
                # Random init with the same std as the original embeddings (default is N(0, 0.02))
                emb[neo_id] = torch.randn_like(emb[0]) * 0.02
            else:
                init_ids = tok(init_word, add_special_tokens=False).input_ids
                init_vec = emb[init_ids].to(torch.float32).mean(0).to(emb.dtype)
                emb[neo_id] = init_vec

        return cls(model, tok, neo_token, neo_id, device)

    def freeze_except_neologism(self):
        """Freeze the whole model; train only the neologism's embedding row."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.emb_weight.requires_grad_(True)
        neo_id = self.neo_id

        def _mask_grad(grad):
            masked = torch.zeros_like(grad)
            masked[neo_id] = grad[neo_id]
            return masked

        self._grad_handle = self.emb_weight.register_hook(_mask_grad)
        return [self.emb_weight]

    @property
    def neo_embedding(self) -> torch.Tensor:
        return self.emb_weight[self.neo_id]

    def embedding_norm(self) -> torch.Tensor:
        return self.neo_embedding.norm()

    # ------------------------------------------------------------- tokenizing
    def tokenize(self, prompt_text: str, response: str, append_eos: bool = True) -> TokenizedExample:
        """Apply the chat template; label only the assistant response tokens.

        The response is whatever should follow the assistant generation prompt
        (for reasoning models this includes the chain-of-thought, which begins
        right after the template's `<think>` marker).
        """
        user_msg = [{"role": "user", "content": prompt_text}]
        prompt_ids = self.tok.apply_chat_template(
            user_msg, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        full_msgs = user_msg + [{"role": "assistant", "content": response}]
        full_ids = self.tok.apply_chat_template(
            full_msgs, add_generation_prompt=False, tokenize=True, return_dict=False
        )
        if not append_eos or full_ids[: len(prompt_ids)] != prompt_ids:
            # template not prefix-consistent (e.g. R1's <think> marker), or we
            # explicitly don't want EOS: fall back to manual concatenation.
            resp_ids = self.tok(response, add_special_tokens=False).input_ids
            eot = self.tok.eos_token_id
            full_ids = prompt_ids + resp_ids + ([eot] if (append_eos and eot is not None) else [])
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return TokenizedExample(input_ids=full_ids, labels=labels)

    def collate(self, examples: list[TokenizedExample]) -> dict[str, torch.Tensor]:
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id
        maxlen = max(len(e.input_ids) for e in examples)
        input_ids, attn, labels = [], [], []
        for e in examples:
            n = len(e.input_ids)
            pad = maxlen - n
            input_ids.append(e.input_ids + [pad_id] * pad)
            attn.append([1] * n + [0] * pad)
            labels.append(e.labels + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, device=self.device),
            "attention_mask": torch.tensor(attn, device=self.device),
            "labels": torch.tensor(labels, device=self.device),
        }

    # ------------------------------------------------------------------ logits
    def _masked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # never allow the neologism to be predicted
        logits = logits.clone()
        logits[..., self.neo_id] = float("-inf")
        return logits

    def nll_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean next-token cross-entropy over response (label != -100) tokens."""
        out = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        logits = self._masked_logits(out.logits)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def sequence_logprob(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-example summed log-prob of the labelled (response) tokens. [B]"""
        out = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        logits = self._masked_logits(out.logits)[:, :-1, :].float()
        labels = batch["labels"][:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        mask = labels != -100
        gather_labels = labels.clamp_min(0).unsqueeze(-1)
        tok_logp = logp.gather(-1, gather_labels).squeeze(-1)
        return (tok_logp * mask).sum(-1)

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> str:
        msgs = [{"role": "user", "content": prompt_text}]
        inputs = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=False,
        ).to(self.device)
        gen = self.model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            suppress_tokens=[self.neo_id],  # never emit the neologism
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        new_tokens = gen[0, inputs.shape[1] :]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def batched_generate(
        self,
        prompt_texts: list[str],
        max_new_tokens: int = 512,
        assistant_prefix: str | None = None,
        **kw,
    ) -> list[str]:
        """Generate completions. If `assistant_prefix` is given (e.g. "To"), the
        assistant turn is seeded with it and the prefix is prepended to each
        returned string -- used to elicit concept-bearing responses that begin a
        particular way.
        """
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        prefix_ids = (
            self.tok(assistant_prefix, add_special_tokens=False).input_ids
            if assistant_prefix
            else []
        )
        seqs = []
        for p in prompt_texts:
            ids = self.tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=True, return_dict=False,
            )
            seqs.append(ids + prefix_ids)

        maxlen = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:  # left-pad
            pad = maxlen - len(s)
            input_ids.append([pad_id] * pad + s)
            attn.append([0] * pad + [1] * len(s))
        input_ids = torch.tensor(input_ids, device=self.device)
        attn = torch.tensor(attn, device=self.device)

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=kw.get("do_sample", False),
            temperature=kw.get("temperature") if kw.get("do_sample") else None,
            top_p=kw.get("top_p") if kw.get("do_sample") else None,
            suppress_tokens=[self.neo_id],
            pad_token_id=pad_id,
        )
        out = []
        for i in range(gen.shape[0]):
            cont = self.tok.decode(gen[i, maxlen:], skip_special_tokens=True)
            text = ((assistant_prefix or "") + cont).strip()
            out.append(text)
        return out

    # --------------------------------------------------------- save / restore
    def save_neologism(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "neo_token": self.neo_token,
                "neo_id": self.neo_id,
                "embedding": self.neo_embedding.detach().to(torch.float32).cpu(),
            },
            path,
        )

    def load_neologism(self, path: str):
        blob = torch.load(path, map_location="cpu")
        with torch.no_grad():
            self.emb_weight[self.neo_id] = blob["embedding"].to(
                self.emb_weight.dtype
            ).to(self.device)
        return blob
