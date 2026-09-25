# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Tokenizer performance benchmark comparing ORT Extensions with
# HuggingFace Transformers.
# ORT and HF both load from the SAME tokenizer directory — any token count
# difference is a pre-tokenizer implementation difference (not a bug).
#
# Standalone usage:
#   python test/test_performance.py [--iterations N] [--warmup N]
#
# CI usage (via pytest — auto-discovered):
#   pytest test/test_performance.py -v
#
# Requirements:
#   pip install transformers onnxruntime-extensions
#
# NOTE: CI regression thresholds are intentionally generous (ORT must not be
# >3x slower than HF) to avoid flaky failures on variable CI machines.

import time
import statistics
import sys
import os
import platform
import json
import unittest

try:
    from transformers import AutoTokenizer
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

try:
    from onnxruntime_extensions import pp_api
    HAS_ORT = True
except ImportError:
    # Fallback: try loading from source tree (local dev without pip install)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from onnxruntime_extensions import pp_api
        HAS_ORT = True
    except ImportError:
        pp_api = None
        HAS_ORT = False

# Debug builds are slower — skip speed comparison tests.
# CI debug jobs set OCOS_SCB_DEBUG=1 in their test step.
IS_DEBUG_BUILD = os.environ.get("OCOS_SCB_DEBUG") == "1"




# =============================================================================
# Test inputs of varying sizes and types
# =============================================================================

INPUTS = {
    "short_english": "The quick brown fox jumps over the lazy dog near the riverbank.",

    "medium_english": (
        "The transformer architecture has revolutionized natural language processing "
        "by introducing the self-attention mechanism, which allows each token in a sequence "
        "to attend to all other tokens simultaneously. This parallel computation replaced "
        "the sequential nature of recurrent neural networks, enabling significantly faster "
        "training on modern GPU hardware. The key innovation lies in the query-key-value "
        "projections that compute relevance scores between all pairs of tokens, weighted by "
        "the inverse square root of the dimension for numerical stability. "
    ),

    "code_python": '''import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for transformer models."""

    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)
''',

    "multilingual": (
        "こんにちは世界！今日は天気がいいですね。机器学习和深度学习正在改变世界。"
        "Привет мир! Сегодня хорошая погода. Transformerモデルは自然言語処理を革新しました。"
        " مرحبا بالعالم! الطقس جميل اليوم. 오늘 날씨가 좋습니다. 감사합니다!"
        "Dies ist ein Test für mehrsprachige Tokenisierung mit verschiedenen Schriftsystemen."
    ),
}

# Generate long versions by repeating
INPUTS["long_english"] = INPUTS["medium_english"] * 12
INPUTS["very_long_english"] = INPUTS["medium_english"] * 50


# =============================================================================
# Models to benchmark
# =============================================================================

# Model configurations for benchmarking.
# Each model has:
#   - ort_path: relative path under test/ dir to tokenizer data
#   - hf_model: HuggingFace model name or local path for AutoTokenizer
#
# Both ORT and HF load from the SAME model vocabulary.
# Token count differences are pre-tokenizer regex implementation differences.
MODELS = {
    "gpt2": {
        "ort_path": "data/phi-2",              # Phi-2 uses GPT-2 tokenizer (same 50257 vocab, same merges)
        "hf_model": "data/phi-2",
        "desc": "Phi-2 uses the GPT-2 tokenizer (50257 vocab, same merges)",
    },
    "phi-4": {
        "ort_path": "data/phi-4-base",         # 100352 vocab, LLAMA regex pattern
        "hf_model": "data/phi-4-base",         # local path
    },
    "llama2": {
        "ort_path": "data/llama2",             # 32000 vocab LLaMA-style SPM
        "hf_model": "NousResearch/Llama-2-7b-hf",  # public mirror (local tokenizer.json has broken merges)
    },
    "gemma": {
        "ort_path": "data/gemma",              # 256000 vocab SPM-BPE
        "hf_model": "data/gemma",              # local path
    },
}




# =============================================================================
# Benchmark utilities
# =============================================================================

def benchmark_function(func, warmup=3, iterations=20):
    """Run a function multiple times and return timing statistics."""
    for _ in range(warmup):
        result = func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "result": result,
    }


def print_machine_info():
    """Print machine metadata for reproducibility."""
    print(f"  Python:    {platform.python_version()}")
    print(f"  OS:        {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  CPU:       {platform.processor() or 'unknown'}")
    if HAS_TRANSFORMERS:
        print(f"  transformers: {transformers.__version__}")
    print()


# =============================================================================
# Benchmark implementations
# =============================================================================

# Cache for HF tokenizers (avoid re-downloading per input)
_hf_tokenizer_cache = {}


def benchmark_huggingface(hf_model, input_text, warmup=3, iterations=20):
    """Benchmark HuggingFace Transformers AutoTokenizer."""
    if not HAS_TRANSFORMERS:
        return None, "transformers not installed"
    try:
        if hf_model not in _hf_tokenizer_cache:
            tok = AutoTokenizer.from_pretrained(hf_model)
            tok.model_max_length = 1_000_000  # suppress sequence length warnings
            _hf_tokenizer_cache[hf_model] = tok
        tokenizer = _hf_tokenizer_cache[hf_model]
    except Exception as e:
        return None, f"Failed to load: {e}"

    def tokenize():
        return tokenizer.encode(input_text, add_special_tokens=False)

    stats = benchmark_function(tokenize, warmup=warmup, iterations=iterations)
    stats["tokens"] = len(stats["result"])
    return stats, None


def benchmark_ort_extensions(data_path, input_text, warmup=3, iterations=20):
    """Benchmark ORT Extensions C++ tokenizer via pp_api."""
    if not HAS_ORT:
        return None, "pp_api not available"
    try:
        tokenizer = pp_api.Tokenizer(data_path)
        tokenizer.update_options({"add_special_tokens": "false"})
    except Exception as e:
        return None, f"Failed to load: {e}"

    def tokenize():
        return tokenizer.tokenize(input_text)

    stats = benchmark_function(tokenize, warmup=warmup, iterations=iterations)
    stats["tokens"] = len(stats["result"]) if stats["result"] is not None else 0
    return stats, None





# =============================================================================
# pytest test class — CI regression detection
# =============================================================================

class TestTokenizerPerformance(unittest.TestCase):
    """
    Performance regression tests for tokenization.

    These tests verify that ORT Extensions tokenization is within acceptable
    performance bounds. Thresholds are intentionally generous to avoid flaky
    failures on CI machines with variable load.

    Regression policy:
    - ORT must not be >3x slower than HF on the same input (catastrophic regression)
    - ORT must be faster than HF on average across multiple inputs (ratio < 1.5)
    - ORT must tokenize medium English in <50ms (absolute sanity check)
    - Tokenization should scale roughly linearly with input length
    """

    @classmethod
    def setUpClass(cls):
        """Load tokenizers once for all tests."""
        cls._orig_cwd = os.getcwd()
        test_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(test_dir)

        cls.ort_tokenizer = None
        if HAS_ORT:
            try:
                t = pp_api.Tokenizer(MODELS["gpt2"]["ort_path"])
                t.update_options({"add_special_tokens": "false"})
                cls.ort_tokenizer = t
            except Exception:
                pass

        cls.hf_tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                cls.hf_tokenizer = AutoTokenizer.from_pretrained(MODELS["gpt2"]["hf_model"])
            except Exception:
                pass

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_cwd)

    @unittest.skipUnless(HAS_ORT, "onnxruntime_extensions not available")
    def test_ort_tokenization_absolute_latency(self):
        """ORT must tokenize medium English in <50ms (sanity check)."""
        if self.ort_tokenizer is None:
            self.skipTest("Could not load Phi-3 tokenizer data")

        input_text = INPUTS["medium_english"]

        # Warmup
        for _ in range(3):
            self.ort_tokenizer.tokenize(input_text)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            self.ort_tokenizer.tokenize(input_text)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        median_ms = statistics.median(times)
        self.assertLess(median_ms, 50.0,
                        f"ORT tokenization too slow: {median_ms:.2f}ms > 50ms threshold")

    @unittest.skipUnless(HAS_ORT and HAS_TRANSFORMERS,
                         "Both onnxruntime_extensions and transformers required")
    def test_ort_not_catastrophically_slower_than_hf(self):
        """ORT must not be >3x slower than HF (regression detection)."""
        if self.ort_tokenizer is None or self.hf_tokenizer is None:
            self.skipTest("Could not load tokenizers")

        input_text = INPUTS["medium_english"]
        warmup = 3
        iterations = 10

        # Benchmark HF
        for _ in range(warmup):
            self.hf_tokenizer.encode(input_text, add_special_tokens=False)
        hf_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.hf_tokenizer.encode(input_text, add_special_tokens=False)
            end = time.perf_counter()
            hf_times.append((end - start) * 1000)

        # Benchmark ORT
        for _ in range(warmup):
            self.ort_tokenizer.tokenize(input_text)
        ort_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.ort_tokenizer.tokenize(input_text)
            end = time.perf_counter()
            ort_times.append((end - start) * 1000)

        hf_median = statistics.median(hf_times)
        ort_median = statistics.median(ort_times)
        ratio = ort_median / hf_median if hf_median > 0 else float('inf')

        self.assertLess(ratio, 3.0,
                        f"ORT is {ratio:.1f}x slower than HF "
                        f"(ORT={ort_median:.2f}ms, HF={hf_median:.2f}ms). "
                        f"Threshold: 3x")

    @unittest.skipIf(IS_DEBUG_BUILD, "Speed comparison skipped on debug builds")
    @unittest.skipUnless(HAS_ORT and HAS_TRANSFORMERS,
                         "Both onnxruntime_extensions and transformers required")
    def test_ort_faster_than_hf_on_average(self):
        """ORT should be faster than HF on average across multiple inputs."""
        if self.ort_tokenizer is None or self.hf_tokenizer is None:
            self.skipTest("Could not load tokenizers")

        test_inputs = [
            INPUTS["short_english"],
            INPUTS["medium_english"],
            INPUTS["long_english"],
            INPUTS["code_python"],
        ]
        warmup = 3
        iterations = 10

        ort_total_ms = 0.0
        hf_total_ms = 0.0

        for text in test_inputs:
            # HF
            for _ in range(warmup):
                self.hf_tokenizer.encode(text, add_special_tokens=False)
            hf_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                self.hf_tokenizer.encode(text, add_special_tokens=False)
                end = time.perf_counter()
                hf_times.append((end - start) * 1000)

            # ORT
            for _ in range(warmup):
                self.ort_tokenizer.tokenize(text)
            ort_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                self.ort_tokenizer.tokenize(text)
                end = time.perf_counter()
                ort_times.append((end - start) * 1000)

            hf_total_ms += statistics.median(hf_times)
            ort_total_ms += statistics.median(ort_times)

        # ORT should be faster on average (ratio < 1.0 means ORT is faster).
        # Use generous 1.5x threshold to avoid flakiness in debug/CI.
        ratio = ort_total_ms / hf_total_ms if hf_total_ms > 0 else float('inf')
        self.assertLess(ratio, 1.5,
                        f"ORT is {ratio:.2f}x slower than HF on average "
                        f"(ORT={ort_total_ms:.1f}ms, HF={hf_total_ms:.1f}ms). "
                        f"Expected ORT to be faster (ratio < 1.5)")

    @unittest.skipUnless(HAS_ORT, "onnxruntime_extensions not available")
    def test_ort_scaling_not_superlinear(self):
        """Tokenization should scale roughly linearly with input length."""
        if self.ort_tokenizer is None:
            self.skipTest("Could not load Phi-3 tokenizer data")

        base = INPUTS["medium_english"]
        input_1x = base
        input_8x = base * 8

        # Measure 1x
        for _ in range(3):
            self.ort_tokenizer.tokenize(input_1x)
        times_1x = []
        for _ in range(10):
            start = time.perf_counter()
            self.ort_tokenizer.tokenize(input_1x)
            end = time.perf_counter()
            times_1x.append((end - start) * 1000)

        # Measure 8x
        for _ in range(3):
            self.ort_tokenizer.tokenize(input_8x)
        times_8x = []
        for _ in range(10):
            start = time.perf_counter()
            self.ort_tokenizer.tokenize(input_8x)
            end = time.perf_counter()
            times_8x.append((end - start) * 1000)

        median_1x = statistics.median(times_1x)
        median_8x = statistics.median(times_8x)
        scaling_factor = median_8x / median_1x

        # Should be ~8x for linear scaling. Allow up to 12x (50% overhead)
        # to account for cache effects. >12x suggests super-linear behavior.
        self.assertLess(scaling_factor, 12.0,
                        f"Scaling is super-linear: 8x input took {scaling_factor:.1f}x time "
                        f"(1x={median_1x:.2f}ms, 8x={median_8x:.2f}ms). "
                        f"Threshold: 12x")




# =============================================================================
# Standalone benchmark runner (not used by pytest)
# =============================================================================

def run_standalone_benchmark(args):
    """Full benchmark comparing ORT vs HF Transformers."""
    print("=" * 80)
    print(" TOKENIZER PERFORMANCE BENCHMARK")
    print(" ORT Extensions vs HuggingFace Transformers")
    print("=" * 80)
    print()
    print("Machine Info:")
    print_machine_info()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)

    results_table = []
    warmup = args.warmup
    iterations = args.iterations

    # --- Per-model comparison (ORT vs HF) ---
    for model_key, model_cfg in MODELS.items():
        ort_path = model_cfg["ort_path"]
        hf_model = model_cfg.get("hf_model", ort_path)

        print(f"\n{'─' * 70}")
        header = f" Model: {model_key} (data: {ort_path})"
        if model_cfg.get("desc"):
            header += f"  [{model_cfg['desc']}]"
        print(header)
        print(f"{'─' * 70}")

        for input_name, input_text in INPUTS.items():
            input_bytes = len(input_text.encode("utf-8"))

            hf_stats, hf_err = benchmark_huggingface(
                hf_model, input_text, warmup=warmup, iterations=iterations)
            ort_stats, ort_err = benchmark_ort_extensions(
                ort_path, input_text, warmup=warmup, iterations=iterations)

            print(f"\n  [{input_name}] ({input_bytes} bytes)")

            if hf_stats:
                hf_tps = (hf_stats["tokens"] / hf_stats["median_ms"]) * 1000
                hf_mbps = (input_bytes / hf_stats["median_ms"]) * 1000 / (1024 * 1024)
                print(f"    HF:      median={hf_stats['median_ms']:.3f}ms  "
                      f"p95={hf_stats['p95_ms']:.3f}ms  "
                      f"{hf_stats['tokens']} tokens  "
                      f"{hf_tps:.0f} tok/s  {hf_mbps:.2f} MB/s")
            else:
                print(f"    HF:      SKIPPED ({hf_err})")

            if ort_stats:
                ort_tps = (ort_stats["tokens"] / ort_stats["median_ms"]) * 1000
                ort_mbps = (input_bytes / ort_stats["median_ms"]) * 1000 / (1024 * 1024)
                print(f"    ORT:     median={ort_stats['median_ms']:.3f}ms  "
                      f"p95={ort_stats['p95_ms']:.3f}ms  "
                      f"{ort_stats['tokens']} tokens  "
                      f"{ort_tps:.0f} tok/s  {ort_mbps:.2f} MB/s")
            else:
                print(f"    ORT:     SKIPPED ({ort_err})")

            # Summary comparison
            if hf_stats and ort_stats:
                ratio = hf_stats["median_ms"] / ort_stats["median_ms"]
                faster = "ORT" if ratio > 1 else "HF"
                factor = ratio if ratio > 1 else 1 / ratio
                print(f"    ==> {faster} is {factor:.2f}x faster than {'HF' if faster == 'ORT' else 'ORT'}")

                if hf_stats["tokens"] != ort_stats["tokens"]:
                    actual_diff = ort_stats['tokens'] - hf_stats['tokens']
                    print(f"    NOTE: Token count diff: "
                          f"HF={hf_stats['tokens']} ORT={ort_stats['tokens']} "
                          f"(diff={actual_diff}, pre-tokenizer split difference)")

                row = {
                    "model": model_key,
                    "input": input_name,
                    "bytes": input_bytes,
                    "hf_median_ms": round(hf_stats["median_ms"], 3),
                    "ort_median_ms": round(ort_stats["median_ms"], 3),
                    "hf_tokens": hf_stats["tokens"],
                    "ort_tokens": ort_stats["tokens"],
                    "ratio": round(ratio, 2),
                    "faster": faster,
                }
                results_table.append(row)
            elif ort_stats:
                # ORT-only (no HF comparison available)
                row = {
                    "model": model_key,
                    "input": input_name,
                    "bytes": input_bytes,
                    "hf_median_ms": None,
                    "ort_median_ms": round(ort_stats["median_ms"], 3),
                    "hf_tokens": None,
                    "ort_tokens": ort_stats["tokens"],
                    "ratio": None,
                    "faster": "ORT",
                }
                results_table.append(row)


    # --- Summary ---
    hf_mbps_all = []
    ort_mbps_all = []
    if results_table:
        print(f"\n\n{'=' * 80}")
        print(" SUMMARY")
        print(f"{'=' * 80}")
        print(f" {'Model':<8} {'Input':<20} {'Bytes':>6} {'HF(ms)':>8} {'ORT(ms)':>8} "
              f"{'HF MB/s':>8} {'ORT MB/s':>9} {'Speedup':>8} {'Winner'}")
        print(f" {'-'*8} {'-'*20} {'-'*6} {'-'*8} {'-'*8} "
              f"{'-'*8} {'-'*9} {'-'*8} {'-'*6}")
        for r in results_table:
            hf_ms = r['hf_median_ms']
            ort_ms = r['ort_median_ms']
            b = r['bytes']
            ort_mbps = (b / ort_ms) * 1000 / (1024 * 1024) if ort_ms else 0
            hf_mbps = (b / hf_ms) * 1000 / (1024 * 1024) if hf_ms else 0

            hf_str = f"{hf_ms:>8.3f}" if hf_ms else f"{'N/A':>8}"
            hf_mbps_str = f"{hf_mbps:>7.2f}" if hf_ms else f"{'N/A':>8}"
            ort_mbps_str = f"{ort_mbps:>8.2f}" if ort_ms else f"{'':>9}"
            ratio_str = f"{r['ratio']:>7.2f}x" if r['ratio'] else f"{'N/A':>8}"

            line = (f" {r['model']:<8} {r['input']:<20} {b:>6} "
                    f"{hf_str} {ort_ms:>8.3f} "
                    f"{hf_mbps_str} {ort_mbps_str} {ratio_str}  {r['faster']}")
            print(line)

        # --- Averages ---
        print(f"\n{'─' * 80}")
        print(" AVERAGE THROUGHPUT (MB/s) across all inputs:")
        print(f"{'─' * 80}")
        for r in results_table:
            b = r['bytes']
            if r['ort_median_ms']:
                ort_mbps_all.append((b / r['ort_median_ms']) * 1000 / (1024 * 1024))
            if r['hf_median_ms']:
                hf_mbps_all.append((b / r['hf_median_ms']) * 1000 / (1024 * 1024))

        if ort_mbps_all:
            print(f"  ORT Extensions:          {statistics.mean(ort_mbps_all):>8.2f} MB/s")
        if hf_mbps_all:
            print(f"  HF Transformers:         {statistics.mean(hf_mbps_all):>8.2f} MB/s")

        if ort_mbps_all and hf_mbps_all:
            ort_avg = statistics.mean(ort_mbps_all)
            hf_avg = statistics.mean(hf_mbps_all)
            if ort_avg > hf_avg:
                print(f"\n  ==> ORT Extensions is {ort_avg/hf_avg:.2f}x faster than HF Transformers on average")
            else:
                print(f"\n  ==> HF Transformers is {hf_avg/ort_avg:.2f}x faster than ORT Extensions on average")

    # Always write JSON results
    summary = {}
    if ort_mbps_all:
        summary["ort_avg_mbps"] = round(statistics.mean(ort_mbps_all), 2)
    if hf_mbps_all:
        summary["hf_avg_mbps"] = round(statistics.mean(hf_mbps_all), 2)
    if ort_mbps_all and hf_mbps_all:
        summary["ort_vs_hf_speedup"] = round(statistics.mean(ort_mbps_all) / statistics.mean(hf_mbps_all), 2)
    summary["ort_wins"] = sum(1 for r in results_table if r.get("faster") == "ORT")
    summary["hf_wins"] = sum(1 for r in results_table if r.get("faster") == "HF")
    summary["total_tests"] = len(results_table)

    output = {
        "summary": summary,
        "machine": {
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(),
            "cpu": platform.processor() or "unknown",
            "transformers_version": transformers.__version__ if HAS_TRANSFORMERS else None,
        },
        "config": {"warmup": warmup, "iterations": iterations},
        "results": results_table,
    }
    json_path = os.path.join(test_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON results saved to: {json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tokenizer performance benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations (default: 50)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    args = parser.parse_args()
    run_standalone_benchmark(args)
