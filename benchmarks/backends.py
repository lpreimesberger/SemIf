"""Backend selection shared by the systems benchmarks (torch/CUDA or llamacpp/GGUF)."""

from __future__ import annotations

from types import SimpleNamespace


def add_arguments(parser) -> None:
    parser.add_argument("--backend", choices=("torch", "llamacpp"), default="torch")
    parser.add_argument("--gguf-file", help="GGUF filename inside a remote --model repo (llamacpp)")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.5-4B",
                        help="Source model whose tokenizer matches the GGUF (llamacpp)")
    parser.add_argument("--tokenizer-revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="llamacpp layers to offload; -1 is all")
    parser.add_argument("--hardware-label", help="Required for llamacpp: the device name recorded in the report")


def validate(parser, args) -> None:
    if args.backend == "torch":
        if args.gguf_file or args.hardware_label:
            parser.error("--gguf-file and --hardware-label require --backend llamacpp")
    else:
        if not args.hardware_label:
            parser.error("--hardware-label is required with --backend llamacpp (llama.cpp does not report a device name)")
        if args.n_gpu_layers < -1:
            parser.error("--n-gpu-layers must be -1 or nonnegative")


def load(args, max_tokens: int = 4096) -> SimpleNamespace:
    """Load the model and return its scorers plus device and peak-memory hooks."""
    if args.backend == "llamacpp":
        from semif_phase1 import llamacpp_backend as backend

        model, tokenizer, metadata = backend.load_model(
            args.model, args.revision, gguf_file=args.gguf_file,
            tokenizer_source=args.tokenizer_model, tokenizer_revision=args.tokenizer_revision,
            n_gpu_layers=args.n_gpu_layers, n_ctx=max_tokens)
        return SimpleNamespace(
            name="llamacpp", model=model, tokenizer=tokenizer, metadata=metadata,
            score=backend.score, SerialPrefixScorer=backend.SerialPrefixScorer, score_shared=backend.score_shared,
            hardware=args.hardware_label, reset_peak=lambda: None,
            peak_key="peak_device_bytes", peak=lambda: None)
    import torch

    from semif_phase1.core import load_causal_model
    from semif_phase1.direct import score
    from semif_phase1.serial import SerialPrefixScorer
    from semif_phase1.shared import score_shared

    model, tokenizer, metadata = load_causal_model(args.model, args.revision)
    return SimpleNamespace(
        name="torch", model=model, tokenizer=tokenizer, metadata=metadata,
        score=score, SerialPrefixScorer=SerialPrefixScorer, score_shared=score_shared,
        hardware=torch.cuda.get_device_name(0), reset_peak=torch.cuda.reset_peak_memory_stats,
        peak_key="peak_cuda_bytes", peak=torch.cuda.max_memory_allocated)
