# AMD and other non-CUDA GPUs / llama.cpp

The `llamacpp` backend runs SemIf's direct, serial-prefix, and shared-state
decision modes on any GPU llama.cpp supports (Vulkan, ROCm, Metal, CPU). It was
developed on an AMD Ryzen AI Max+ 395 (Radeon 8060S, Vulkan/RADV). Like the
Torch and MLX backends it reads last-position logits for the declared answer
slots; no token is generated. Scores are conditional option scores, not
calibrated confidence.

## Install

llama-cpp-python must be compiled for your GPU. For Vulkan on Ubuntu:

```bash
sudo apt install libvulkan-dev glslc spirv-headers vulkan-tools
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
CMAKE_ARGS='-DGGML_VULKAN=on' pip install llama-cpp-python --no-cache-dir
```

Your user needs access to `/dev/dri/renderD*` (the `render` and `video` groups).
`vulkaninfo --summary` should list your GPU. If the build has no GPU offload the
loader fails instead of silently running on CPU; pass `--n-gpu-layers 0` to
choose CPU on purpose.

## Score

```bash
semif-score --backend llamacpp --mode direct \
  --model bartowski/Qwen_Qwen3.5-4B-GGUF \
  --revision 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 \
  --gguf-file Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --input examples/decisions.jsonl --output results-llamacpp.jsonl
```

`--model` may instead be a local `.gguf` file with any revision label. Prompts
are rendered and tokenized by the pinned source model's Hugging Face tokenizer
(`--tokenizer-model`, `--tokenizer-revision`; default Qwen3.5-4B). Every prompt
is re-tokenized by llama.cpp and must match exactly, so a GGUF whose vocabulary
differs from the tokenizer is rejected. Results record the GGUF SHA-256, file
type, llama-cpp-python version, and offload settings.

`--mode serial` and `--mode shared` snapshot llama.cpp state after the state
prefix and restore it per question, because Qwen3.5's recurrent layers cannot be
trimmed to an arbitrary position. Shared mode runs suffixes one after another
rather than as a padded batch. Each restore copies the saved state (about
112 MB for a 1,800-token Qwen3.5-4B prefix, roughly 0.1 s on Strix Halo) and its
cost does not depend on `--max-tokens`. That restore, not the suffix forward
pass, dominates reuse time. Reranker mode is unsupported.

## Measured on Strix Halo (Q4_K_M, Vulkan)

Quantization changes probabilities; these are not the BF16 headline numbers.

- Authored 144 decisions, serial: 138/144 argmaxes match the published BF16 Torch
  run; raw accuracy 0.792 vs 0.806. Median largest probability difference 0.016
  (max 0.382). Minimum allowed-token mass 0.979.
- One 1,812-token state with 21 questions: shared and serial argmaxes match
  direct on 21/21 (largest probability difference 0.039). Direct 23.9 s;
  serial 5.8 s; shared 5.8 s (1.9 s prefill, 1.9 s state restores).

- All 777 decisions (`benchmarks/shape777.py`, 37 states x 21 questions):

  | Mode | Wall time | Decisions/s |
  |---|---:|---:|
  | Fresh direct | 897 s | 0.87 |
  | Serial prefix reuse | 213 s | 3.66 |
  | Shared state | 209 s | 3.71 |

  Serial and shared each changed 4 of 777 argmaxes relative to fresh (largest
  probability difference 0.073). Shared is not meaningfully faster than serial
  here because it also runs suffixes sequentially.
- Compact generation baseline (`benchmarks/decision_vs_generation.py`): greedy
  generation of the 21-value array took 3.1 s and was invalid in every run (the
  Q4_K_M model wrote 20 values), against 5.7 s for shared direct scoring. The
  README's speed advantage over generation, measured on CUDA with a batched
  parallel path, does not carry over to this backend.

These are single local runs, not committed evidence under `results/`.

## Systems benchmarks

`benchmarks/shape777.py` and `benchmarks/decision_vs_generation.py` accept the
same backend flags. llama.cpp does not report a device name, so pass
`--hardware-label`; memory is recorded as `peak_device_bytes: null` because
llama.cpp exposes no peak-allocation counter comparable to CUDA's.

```bash
python benchmarks/shape777.py --backend llamacpp \
  --model bartowski/Qwen_Qwen3.5-4B-GGUF \
  --revision 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 \
  --gguf-file Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --hardware-label "AMD Radeon 8060S (Strix Halo, Vulkan)" \
  --input benchmarks/data/shape777.jsonl --output shape777-llamacpp.json
```

The generation comparison uses greedy llama.cpp generation. Quality evaluation
needs no flags: score with `semif-score --backend llamacpp` and pass the
predictions to `benchmarks/evaluate.py` as usual.

## Tests

`pytest -q` validates loader inputs without weights. To also run the real-GPU
regression: `SEMIF_TEST_GGUF=/path/to/model.gguf pytest -q`.
