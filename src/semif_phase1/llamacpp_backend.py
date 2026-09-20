"""GGUF option readout through llama.cpp (Vulkan/ROCm/CPU), e.g. AMD GPUs without CUDA.

Token IDs come from the model's Hugging Face tokenizer and are verified against
llama.cpp's own tokenization of every prompt, so a mismatched GGUF vocabulary
fails loudly. Scores are last-position logits restricted to the declared answer
slots; no token is generated. They remain conditional option scores, not
calibrated confidence. Prefix reuse snapshots llama.cpp state, because Qwen3.5's
recurrent layers cannot be trimmed to an arbitrary position.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .core import LETTERS, direct_messages, softmax
from .direct import PROMPT_VERSION, encode_prompt
from .serial import _state_prefix

DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def load_model(source: str, revision: str, *, gguf_file: str | None = None,
               tokenizer_source: str = DEFAULT_TOKENIZER_MODEL,
               tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
               n_gpu_layers: int = -1, n_ctx: int = 4096):
    """Load a pinned GGUF plus its pinned source tokenizer.

    `source` is a local .gguf file, or a Hugging Face GGUF repo id with a
    40-character `revision` and `gguf_file`. Custom tokenizer code is prohibited.
    """
    if type(n_ctx) is not int or n_ctx < 1 or type(n_gpu_layers) is not int or n_gpu_layers < -1:
        raise ValueError("n_ctx must be positive and n_gpu_layers must be -1 or nonnegative")
    local = Path(source).is_file()
    if not revision or (not local and not _COMMIT.fullmatch(revision)):
        raise ValueError("Remote GGUF repos require a pinned 40-character revision; local files require a revision label")
    tokenizer_local = Path(tokenizer_source).is_dir()
    if not tokenizer_local and not _COMMIT.fullmatch(tokenizer_revision or ""):
        raise ValueError("Remote tokenizers require a pinned 40-character revision")
    try:
        import llama_cpp
    except ImportError as error:
        raise RuntimeError("Install llama-cpp-python built for your GPU, e.g. "
                           "CMAKE_ARGS='-DGGML_VULKAN=on' pip install llama-cpp-python") from error
    import transformers

    if n_gpu_layers != 0 and not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("This llama-cpp-python build has no GPU offload; rebuild with "
                           "-DGGML_VULKAN=on (or pass --n-gpu-layers 0 to run on CPU)")
    if local:
        path = Path(source)
    else:
        if not gguf_file:
            raise ValueError("Remote GGUF repos require --gguf-file")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(source, gguf_file, revision=revision))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source, revision=None if tokenizer_local else tokenizer_revision,
        local_files_only=tokenizer_local, trust_remote_code=False)
    artifact_sha256 = _sha256(path)
    llm = llama_cpp.Llama(model_path=str(path), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
                          logits_all=False, embedding=False, verbose=False)
    metadata = {
        "source": source, "revision": revision, "backend": "llamacpp", "gguf_file": path.name,
        "gguf_sha256": artifact_sha256,
        "gguf_file_type": llm.metadata.get("general.file_type"),
        "gguf_architecture": llm.metadata.get("general.architecture"),
        "tokenizer_source": tokenizer_source, "tokenizer_revision": None if tokenizer_local else tokenizer_revision,
        "llama_cpp_python_version": llama_cpp.__version__,
        "transformers_version": transformers.__version__,
        "n_gpu_layers": n_gpu_layers, "n_ctx": n_ctx, "gpu_offload": bool(llama_cpp.llama_supports_gpu_offload()),
    }
    return llm, tokenizer, metadata


def _encode(llm, tokenizer, row, max_tokens):
    """Encode with the HF tokenizer, then prove llama.cpp tokenizes the prompt identically."""
    encoded = encode_prompt(tokenizer, row, max_tokens)
    ids, slots, _ = encoded
    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if llm.tokenize(prompt.encode(), add_bos=False, special=True) != ids:
        raise ValueError(f"Row {row['id']}: GGUF tokenization differs from the source tokenizer")
    for letter, token in zip(LETTERS, slots):
        if llm.detokenize([token]) != letter.encode():
            raise ValueError(f"Answer slot {letter!r} is not token {token} in the GGUF vocabulary")
    return encoded


def _last_logits(llm) -> np.ndarray:
    # llama-cpp-python only fills `scores` when logits_all is on, which would compute
    # the full vocabulary head for every prompt token; read the last position directly.
    return np.array(np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)), dtype=np.float32)


def _result(row, encoded, logits, metadata, mode):
    ids, slots, prompt_hash = encoded
    selected = logits[slots].astype(np.float64)
    return {
        "id": row["id"], "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected.tolist()), "option_logits": selected.tolist(),
        "answer_token_ids": slots, "input_tokens": len(ids),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "prompt_sha256": prompt_hash, "prompt_version": PROMPT_VERSION,
        "allowed_token_mass": float(np.exp(np.logaddexp.reduce(selected) - np.logaddexp.reduce(logits.astype(np.float64)))),
        "full_vocab_argmax_id": int(logits.argmax()),
        "model": {**metadata, "serving_config": f"llamacpp-{mode}-v1"},
        "readout": "native last-position logits restricted to declared answer slots; no generated tokens",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def _check_prefix(prefix, encoded):
    if not prefix or any(ids[:len(prefix)] != prefix or len(ids) <= len(prefix) for ids, _, _ in encoded):
        raise ValueError("The fixed state prefix does not match every full prompt")


def _prefill(llm, prefix):
    llm.reset()
    llm.eval(prefix)
    return llm.save_state()


def score(llm, tokenizer, row, metadata, max_tokens=4096):
    started = time.perf_counter()
    encoded = _encode(llm, tokenizer, row, max_tokens)
    mark = time.perf_counter()
    llm.reset()
    llm.eval(encoded[0])
    result = _result(row, encoded, _last_logits(llm), metadata, "direct")
    result.update(forward_seconds=time.perf_counter() - mark, total_seconds=time.perf_counter() - started)
    return result


class SerialPrefixScorer:
    """Reuse only the current exact state; every question restores the saved prefix."""

    def __init__(self, llm, tokenizer, metadata, max_tokens=4096):
        self.llm, self.tokenizer, self.metadata = llm, tokenizer, metadata
        self.max_tokens = max_tokens
        self.prefix = self.snapshot = None

    def score(self, row):
        started = time.perf_counter()
        encoded = _encode(self.llm, self.tokenizer, row, self.max_tokens)
        prefix = _state_prefix(self.tokenizer, row["state"])
        _check_prefix(prefix, [encoded])
        hit = self.snapshot is not None and prefix == self.prefix
        prefill_seconds = 0.0
        if not hit:
            self.snapshot = self.prefix = None
            mark = time.perf_counter()
            self.snapshot = _prefill(self.llm, prefix)
            self.prefix = prefix
            prefill_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        self.llm.load_state(self.snapshot)
        restore_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        self.llm.eval(encoded[0][len(prefix):])
        suffix_seconds = time.perf_counter() - mark
        result = _result(row, encoded, _last_logits(self.llm), self.metadata, "serial")
        result.update(cache_hit=hit, prefix_tokens=len(prefix), prefill_seconds=prefill_seconds,
                      copy_seconds=restore_seconds, suffix_forward_seconds=suffix_seconds,
                      forward_seconds=prefill_seconds + suffix_seconds,
                      total_seconds=time.perf_counter() - started)
        return result


def score_shared(llm, tokenizer, rows, metadata, max_tokens=4096):
    """Prefill one exact state once, then score each question from the saved state.

    Suffixes run one after another, not as a padded batch, so nothing is padded
    and no branch can see another's tokens.
    """
    if not rows or any(row["state"] != rows[0]["state"] for row in rows):
        raise ValueError("Shared scoring requires one nonempty exact state")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    started = time.perf_counter()
    encoded = [_encode(llm, tokenizer, row, max_tokens) for row in rows]
    prefix = _state_prefix(tokenizer, rows[0]["state"])
    _check_prefix(prefix, encoded)
    encode_seconds = time.perf_counter() - started
    mark = time.perf_counter()
    snapshot = _prefill(llm, prefix)
    prefill_seconds = time.perf_counter() - mark
    restore_seconds = suffix_seconds = 0.0
    results = []
    for row, enc in zip(rows, encoded):
        mark = time.perf_counter()
        llm.load_state(snapshot)
        restore_seconds += time.perf_counter() - mark
        mark = time.perf_counter()
        llm.eval(enc[0][len(prefix):])
        suffix_seconds += time.perf_counter() - mark
        results.append(_result(row, enc, _last_logits(llm), metadata, "shared"))
    timing = {
        "total_seconds": time.perf_counter() - started, "encode_seconds": encode_seconds,
        "prefix_tokens": len(prefix), "prefill_seconds": prefill_seconds,
        "replicate_seconds": restore_seconds, "suffix_forward_seconds": suffix_seconds,
        "batch_size": len(rows), "true_suffix_tokens": sum(len(ids) - len(prefix) for ids, _, _ in encoded),
        "padded_suffix_tokens": sum(len(ids) - len(prefix) for ids, _, _ in encoded),
    }
    return results, timing
