"""Loader validation always runs; the GPU/GGUF regression runs only when SEMIF_TEST_GGUF is set."""
import json
import os
from pathlib import Path

import pytest

from semif_phase1 import llamacpp_backend as backend


@pytest.mark.parametrize("kwargs,message", [
    (dict(source="repo/x", revision="main", gguf_file="x.gguf"), "40-character revision"),
    (dict(source="repo/x", revision="a" * 40, gguf_file="x.gguf", tokenizer_revision="main"), "tokenizers require a pinned"),
    (dict(source="repo/x", revision="a" * 40, gguf_file="x.gguf", n_ctx=0), "n_ctx must be positive"),
    (dict(source="repo/x", revision="a" * 40, gguf_file="x.gguf", n_gpu_layers=-2), "n_gpu_layers"),
])
def test_loader_rejects_unpinned_or_invalid_inputs(kwargs, message):
    source, revision = kwargs.pop("source"), kwargs.pop("revision")
    with pytest.raises(ValueError, match=message):
        backend.load_model(source, revision, **kwargs)


GGUF = os.environ.get("SEMIF_TEST_GGUF")


@pytest.mark.skipif(not GGUF, reason="set SEMIF_TEST_GGUF to a Qwen3.5 GGUF path to run on the local GPU")
def test_direct_serial_and_shared_agree_on_real_gguf(tmp_path):
    pytest.importorskip("llama_cpp")
    llm, tokenizer, metadata = backend.load_model(GGUF, "local-test", n_ctx=2048)
    rows = [dict(id=str(i), state="The deployment completed at 14:02 UTC. Health checks passed.",
                 question=question, options=[{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}])
            for i, question in enumerate(["Did the deployment succeed?", "Did the deployment fail?"])]
    direct = [backend.score(llm, tokenizer, row, metadata, 2048) for row in rows]
    serial_scorer = backend.SerialPrefixScorer(llm, tokenizer, metadata, 2048)
    serial = [serial_scorer.score(row) for row in rows]
    shared, timing = backend.score_shared(llm, tokenizer, rows, metadata, 2048)
    assert [r["cache_hit"] for r in serial] == [False, True]
    assert timing["batch_size"] == 2
    for a, b, c in zip(direct, serial, shared):
        assert a["allowed_token_mass"] > 0.5
        best = max(range(2), key=a["probabilities"].__getitem__)
        assert max(range(2), key=b["probabilities"].__getitem__) == best
        assert max(range(2), key=c["probabilities"].__getitem__) == best
        assert b["option_logits"] == pytest.approx(a["option_logits"], abs=0.5)
        json.dumps(a, allow_nan=False)


def _benchmark_helper():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "backends.py"
    spec = importlib.util.spec_from_file_location("benchmark_backends", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("argv,message", [
    (["--backend", "llamacpp"], "--hardware-label is required"),
    (["--gguf-file", "x.gguf"], "require --backend llamacpp"),
    (["--backend", "llamacpp", "--hardware-label", "GPU", "--n-gpu-layers", "-2"], "must be -1 or nonnegative"),
])
def test_benchmark_backend_arguments_are_validated(argv, message, capsys):
    import argparse

    helper = _benchmark_helper()
    parser = argparse.ArgumentParser()
    helper.add_arguments(parser)
    with pytest.raises(SystemExit):
        helper.validate(parser, parser.parse_args(argv))
    assert message in capsys.readouterr().err


def test_benchmark_backend_defaults_to_torch():
    import argparse

    helper = _benchmark_helper()
    parser = argparse.ArgumentParser()
    helper.add_arguments(parser)
    args = parser.parse_args([])
    helper.validate(parser, args)
    assert args.backend == "torch"
