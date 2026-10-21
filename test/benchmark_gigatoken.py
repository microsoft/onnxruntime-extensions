"""Matched tokenizer-ID and latency pilot for ORT, Gigatoken, and HF.

Run from the repository root with the editable extension installed:
    python test/benchmark_gigatoken.py

Use --model-dir to select a local tokenizer; --allow-missing-gigatoken runs only the ORT/HF control.
"""

import argparse
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tokenizers import Tokenizer as HFTokenizer
from onnxruntime_extensions import pp_api

from test_performance import INPUTS


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "data" / "phi-2"


def assert_same_ids(reference, candidate, label):
    expected = list(reference)
    actual = list(candidate)
    if expected != actual:
        mismatch = next(
            (index for index, (left, right) in enumerate(zip(expected, actual)) if left != right),
            min(len(expected), len(actual)),
        )
        raise AssertionError(
            f"{label}: first differing ID at {mismatch}; lengths {len(expected)} vs {len(actual)}; "
            f"expected {expected[mismatch:mismatch + 5]}, got {actual[mismatch:mismatch + 5]}"
        )


def median_ms(encode, warmup, iterations):
    for _ in range(warmup):
        encode()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        encode()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def report(label, byte_count, elapsed_ms):
    print(f"{label:34s} {elapsed_ms:9.3f} ms  {byte_count / elapsed_ms / 1000:9.2f} MB/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-gigatoken", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.documents < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("documents and iterations must be positive; warmup cannot be negative")
    model_dir = args.model_dir.resolve()
    if not (model_dir / "tokenizer.json").is_file():
        parser.error(f"missing tokenizer.json in {model_dir}")

    try:
        import gigatoken
    except ModuleNotFoundError as error:
        if error.name != "gigatoken" or not args.allow_missing_gigatoken:
            raise
        gigatoken = None

    ort = pp_api.Tokenizer(str(model_dir))
    ort.update_options({"add_special_tokens": "false"})
    hf = HFTokenizer.from_file(str(model_dir / "tokenizer.json"))
    giga = gigatoken.Tokenizer(str(model_dir)) if gigatoken is not None else None

    cases = {
        **INPUTS,
        "unique_ascii": " ".join(f"row{index}-item{index * 7}" for index in range(2200)),
        "repeated_code": INPUTS["code_python"] * 15,
    }
    print(f"Vocabulary: {model_dir}; add_special_tokens=false")
    print(f"CPU batch cap: RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']}")
    if giga is None:
        print("CONTROL ONLY: Gigatoken unavailable; no Gigatoken comparison follows")

    for case, text in cases.items():
        reference = hf.encode(text, add_special_tokens=False).ids
        assert_same_ids(reference, ort.tokenize(text), f"ORT {case}")
        if giga is not None:
            assert_same_ids(reference, giga.encode(text), f"Gigatoken {case}")
        byte_count = len(text.encode("utf-8"))
        print(f"{case}: {byte_count} bytes, {len(reference)} IDs match")
        if not args.validate_only:
            methods = {
                "ORT single": lambda text=text: ort.tokenize(text),
                "HF single": lambda text=text: hf.encode(text, add_special_tokens=False),
            }
            if giga is not None:
                methods["Gigatoken single"] = lambda text=text: giga.encode(text)
            for name, encode in methods.items():
                report(name, byte_count, median_ms(encode, args.warmup, args.iterations))

    sources = (
        INPUTS["medium_english"],
        INPUTS["code_python"],
        INPUTS["multilingual"],
        INPUTS["very_long_english"],
    )
    documents = [sources[index % len(sources)] + f" batch-id:{index}" for index in range(args.documents)]
    reference_batch = [encoded.ids for encoded in hf.encode_batch(documents, add_special_tokens=False)]
    for index, (reference, candidate) in enumerate(zip(reference_batch, ort.tokenize(documents))):
        assert_same_ids(reference, candidate, f"ORT document {index}")
    if giga is not None:
        for parallel in (False, True):
            candidate_batch = giga.encode_batch_list(documents, parallel=parallel)
            if len(candidate_batch) != len(reference_batch):
                raise AssertionError("Gigatoken changed the number of documents")
            for index, (reference, candidate) in enumerate(zip(reference_batch, candidate_batch)):
                assert_same_ids(reference, candidate, f"Gigatoken document {index}, parallel={parallel}")

    byte_count = sum(len(text.encode("utf-8")) for text in documents)
    print(f"Batch: {len(documents)} documents, {byte_count} bytes, all IDs match")
    if args.validate_only:
        return
    methods = {
        "ORT native batch (sequential)": lambda: ort.tokenize(documents),
        "HF batch (parallelism disabled)": lambda: hf.encode_batch(documents, add_special_tokens=False),
    }
    if giga is not None:
        methods["Gigatoken batch (sequential)"] = lambda: giga.encode_batch_list(documents, parallel=False)
        methods["Gigatoken batch (2 threads)"] = lambda: giga.encode_batch_list(documents, parallel=True)
    for name, encode in methods.items():
        report(name, byte_count, median_ms(encode, args.warmup, args.iterations))


if __name__ == "__main__":
    main()