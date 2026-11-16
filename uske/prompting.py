"""Chat formatting and response post-processing shared by all scripts."""

from __future__ import annotations

import re

from .languages import build_benign_coding_prompt, build_system_prompt, build_task_prompt

# Backbones used in the paper. Any other HF id can be passed directly.
MODEL_MAP = {
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    "qwen2.5-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen3-14b": "Qwen/Qwen3-14B",
}

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*([\s\S]*?)```", re.MULTILINE)


def resolve_model_id(model: str) -> str:
    return MODEL_MAP.get(model, model)


def apply_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Format one system+user exchange with the tokenizer's chat template.

    Falls back to the Llama-2 / CodeLlama instruction format when the tokenizer
    does not ship a chat template.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"


def secure_code_prompt(tokenizer, task: str, lang: str, template_mode: str = "baseline") -> str:
    """Prompt for ProSec / CyberSecEval tasks. `template_mode='secure'` is the PKI baseline."""
    return apply_chat(tokenizer, build_system_prompt(lang, template_mode),
                      build_task_prompt(task, lang, template_mode))


def benign_code_prompt(tokenizer, task: str, lang: str) -> str:
    return apply_chat(tokenizer, build_system_prompt(lang, "baseline"),
                      build_benign_coding_prompt(task, lang))


def extract_code(text: str) -> str:
    """Return the code inside markdown fences if present, otherwise the raw text."""
    if not isinstance(text, str):
        return ""
    blocks = [b.strip() for b in _CODE_BLOCK_RE.findall(text) if b.strip()]
    return "\n\n".join(blocks).strip() if blocks else text.strip()
