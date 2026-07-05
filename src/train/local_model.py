"""Inference wrapper so trained checkpoints (tier1/2/3) plug into the same
loop.py/swarm.py harness frontier models run through — a model string like
"local:qwen2.5-1.5b-tier1-lora-sft" resolves to output/checkpoints/<name>/
and is served locally via transformers instead of litellm.

Single-GPU note: models are cached in-process after first load, but there's
only one GPU (gpu0) behind this — set config.MAX_WORKERS = 1 in src/config.py
before running local models through run_swarm(), otherwise concurrent threads
will try to load multiple checkpoints onto the same device at once.
"""
import json
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config as train_config

LOCAL_PREFIX = "local:"

_lock = threading.Lock()
_cache = {}


def is_local_model(model):
    return model.startswith(LOCAL_PREFIX)


def _load(name):
    with _lock:
        if name in _cache:
            return _cache[name]

        ckpt_dir = train_config.checkpoint_dir(name)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint named {name!r} at {ckpt_dir} "
                f"(expected {meta_path})"
            )
        meta = json.loads(meta_path.read_text())
        base_model_id = meta["base_model"]

        print(f"  [local_model] loading {name} (base={base_model_id}, adapter={meta.get('adapter')})...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
        )

        if meta.get("adapter"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ckpt_dir / "adapter"))

        model.eval()
        _cache[name] = (tokenizer, model)
        return _cache[name]


def local_completion(model, prompt, temperature=0.0, max_tokens=300):
    """Same return shape as loop.py's _call_api, so it's a drop-in for the
    litellm.completion branch."""
    name = model[len(LOCAL_PREFIX):]
    tokenizer, hf_model = _load(name)

    messages = [{"role": "user", "content": prompt}]
    # return_dict=True (the default in this transformers version) so this
    # comes back as a BatchEncoding with input_ids + attention_mask, not a
    # bare tensor — pass it straight through to generate() as kwargs.
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        # Matches the enable_thinking=False used during training (data.py /
        # grpo_train.py) — Qwen3's template defaults to a <think> block
        # otherwise, which the trained checkpoint never saw. Harmlessly
        # ignored by templates (e.g. Qwen2.5's) that don't use this variable.
        enable_thinking=False,
    ).to(hf_model.device)
    input_ids = encoded["input_ids"]

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        output_ids = hf_model.generate(**encoded, **gen_kwargs)

    completion_ids = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    prompt_tokens = int(input_ids.shape[1])
    completion_tokens = int(completion_ids.shape[0])

    return {
        "content": text,
        "reasoning_content": None,
        "finish_reason": "stop",
        "model": model,
        "id": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
