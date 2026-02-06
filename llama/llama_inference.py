"""
Step 7: Llama 4 inference wrapper for this project

Goal:
- Load Llama 4 Scout Instruct via HF Transformers
- Accept chat `messages` (from Step 6) and generate model output
- Return the generated assistant text (expected to be JSON list of {id,value})

Key details:
- Uses AutoProcessor + Llama4ForConditionalGeneration (same pattern as your prior project)
- Uses apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
- Attempts to extract only the newly generated portion (not the prompt), robustly
- Keeps the generation config similar to your known-working config, but parameterized
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Llama4Config:
    model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    cache_dir: str = "../.cache"
    hf_token_env: str = "HF_TOKEN"

    # Tokenization / context window
    max_input_length: int = 8192

    # Generation defaults (tune in ablations)
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9


class Llama4Inference:
    def __init__(self, cfg: Llama4Config) -> None:
        self.cfg = cfg
        self.processor = None
        self.model = None

    def load(self) -> None:
        token = os.environ.get(self.cfg.hf_token_env, None)
        if not token:
            raise RuntimeError(
                f"Missing HF token. Set environment variable {self.cfg.hf_token_env}."
            )

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            cache_dir=self.cfg.cache_dir,
            token=token,
        )

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=self.cfg.cache_dir,
            token=token,
        ).eval()

    @property
    def device(self):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model.device

    def build_chat_text(self, messages: List[Dict[str, Any]]) -> str:
        if self.processor is None:
            raise RuntimeError("Processor not loaded.")
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Run generation and return only the assistant's newly generated text.

        Extraction method:
        - Encode the full chat prompt to input_ids.
        - Generate output_ids.
        - Slice: output_ids[0][len(input_ids[0]):] to get only newly generated tokens.
        - Decode that slice.

        This avoids prompt-echo artifacts and is usually more reliable than decoding whole sequence.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("Call load() before generation.")

        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        temperature = temperature if temperature is not None else self.cfg.temperature
        top_p = top_p if top_p is not None else self.cfg.top_p

        # If temperature is 0, sampling should be disabled for deterministic decoding.
        if do_sample is None:
            do_sample = temperature > 0.0

        chat_text = self.build_chat_text(messages)

        # Tokenize
        inputs = self.processor(
            text=chat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_input_length,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Slice newly generated tokens (assistant continuation)
        gen_ids = output_ids[0, prompt_len:]
        generated_text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return generated_text
