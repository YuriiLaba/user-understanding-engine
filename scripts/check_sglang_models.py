#!/usr/bin/env python3
"""Smoke-test SGLang GPU loading for the project model shortlist.

This intentionally loads one model at a time, runs a tiny generation, then
shuts the engine down so a failure on one model does not stop the rest.
"""

from __future__ import annotations

import argparse
import gc
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelCase:
    label: str
    model_path: str
    quantized: bool = False
    multimodal: bool = False
    gguf_repo: str | None = None
    gguf_file: str | None = None
    gguf_quant: str | None = None


@dataclass(frozen=True)
class CheckResult:
    case: ModelCase
    source: str
    passed: bool
    detail: str


CASES: list[ModelCase] = [
    ModelCase("Qwen-3.5-4b float16", "Qwen/Qwen3.5-4B"),
    ModelCase("Qwen-3.5-4b FP8", "RedHatAI/Qwen3.5-4B-FP8-dynamic", quantized=True),

    ModelCase("Qwen-3.5-9b float16", "Qwen/Qwen3.5-9B"),
    ModelCase("Qwen-3.5-9b FP8", "RedHatAI/Qwen3.5-9B-FP8-dynamic", quantized=True),

    ModelCase("Qwen3-4b float16", "Qwen/Qwen3-4B"),
    ModelCase("Qwen3-4b FP8", "Qwen/Qwen3-4B-FP8", quantized=True),

    ModelCase("Qwen3-8b float16", "Qwen/Qwen3-8B"),
    ModelCase("Qwen3-8b FP8", "Qwen/Qwen3-8B-FP8", quantized=True),

    ModelCase("Qwen3-VL-4b float16", "Qwen/Qwen3-VL-4B-Instruct", multimodal=True),
    ModelCase(
        "Qwen3-VL-4b FP8",
        "Qwen/Qwen3-VL-4B-Instruct-FP8",
        quantized=True,
        multimodal=True,
    ),

    ModelCase("Qwen3-VL-8b float16", "Qwen/Qwen3-VL-8B-Instruct", multimodal=True),
    ModelCase(
        "Qwen3-VL-8b FP8",
        "Qwen/Qwen3-VL-8B-Instruct-FP8",
        quantized=True,
        multimodal=True,
    ),
 
    ModelCase(
        "SmolVlm2-2.2b float16", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", multimodal=True
    ),
    
    ModelCase("Gemma4 float16", "google/gemma-4-E4B-it", multimodal=True),

    ModelCase("LFM2.5-VL-1.6B BF16", "LiquidAI/LFM2.5-VL-1.6B", multimodal=True),
    
    ModelCase("LFM2.5-8B-A1B BF16", "LiquidAI/LFM2.5-8B-A1B"),
    ModelCase("LFM2.5-8B-A1B FP8", "Rifky/LFM2.5-8B-A1B-FP8", quantized=True),
  
    ModelCase("Granite 4.1 8B BF16", "ibm-granite/granite-4.1-8b"),
    ModelCase("Granite 4.1 8B FP8", "ibm-granite/granite-4.1-8b-fp8", quantized=True),

    ModelCase("Granite 4.1 3B BF16", "ibm-granite/granite-4.1-3b"),
    ModelCase("Granite 4.1 3B FP8", "ibm-granite/granite-4.1-3b-fp8", quantized=True),
   
    ModelCase("Nemotron 3 Nano 4B BF16", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"),
    ModelCase(
        "Nemotron 3 Nano 4B FP8", "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8", quantized=True
    ),
]


def resolve_model_path(case: ModelCase, cache_dir: Path | None) -> str:
    if case.gguf_repo:
        from huggingface_hub import HfApi, hf_hub_download

        assert case.gguf_file is not None
        try:
            return hf_hub_download(
                repo_id=case.gguf_repo,
                filename=case.gguf_file,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        except Exception:
            if not case.gguf_quant:
                raise

        candidates = [
            path
            for path in HfApi().list_repo_files(case.gguf_repo)
            if path.endswith(".gguf")
            and case.gguf_quant.lower() in Path(path).name.lower()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No GGUF file containing {case.gguf_quant!r} found in {case.gguf_repo}"
            )
        if len(candidates) > 1:
            candidates = [
                path for path in candidates if "mmproj" not in Path(path).name.lower()
            ] or candidates

        return hf_hub_download(
            repo_id=case.gguf_repo,
            filename=sorted(candidates)[0],
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    return case.model_path


def run_case(case: ModelCase, args: argparse.Namespace) -> tuple[bool, str]:
    import sglang as sgl

    engine = None
    try:
        model_path = resolve_model_path(case, args.cache_dir)
        engine_kwargs = {
            "model_path": model_path,
            "tp_size": args.tp_size,
            "dp_size": args.dp_size,
            "context_length": args.context_length,
            "mem_fraction_static": args.mem_fraction_static,
            "trust_remote_code": args.trust_remote_code,
        }
        if case.multimodal:
            engine_kwargs["limit_mm_data_per_request"] = {"image": 1}

        engine = sgl.Engine(**engine_kwargs)
        output = engine.generate(
            prompt="User: Say OK only.\nAssistant:",
            sampling_params={
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.0,
            },
        )
        return True, str(output)[:500].replace("\n", " ")
    except Exception:
        return False, traceback.format_exc()
    finally:
        if engine is not None:
            engine.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def case_source(case: ModelCase) -> str:
    if case.gguf_repo:
        return f"{case.gguf_repo}/{case.gguf_file}"
    return case.model_path


def one_line_detail(detail: str) -> str:
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    if not lines:
        return ""

    for line in reversed(lines):
        if not line.startswith("File "):
            return line[:180]
    return lines[-1][:180]


def print_result_table(title: str, results: list[CheckResult]) -> None:
    print(f"\n{title} ({len(results)})")
    if not results:
        print("  none")
        return

    status_width = max(
        len("STATUS"), *(len("PASS" if result.passed else "FAIL") for result in results)
    )
    label_width = max(len("MODEL"), *(len(result.case.label) for result in results))
    source_width = max(len("SOURCE"), *(len(result.source) for result in results))

    print(
        f"  {'STATUS':<{status_width}}  {'MODEL':<{label_width}}  {'SOURCE':<{source_width}}  DETAIL"
    )
    print(
        f"  {'-' * status_width}  {'-' * label_width}  {'-' * source_width}  {'-' * 40}"
    )
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  {status:<{status_width}}  "
            f"{result.case.label:<{label_width}}  "
            f"{result.source:<{source_width}}  "
            f"{one_line_detail(result.detail)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", default="", help="Case label substring filter, case-insensitive."
    )
    parser.add_argument("--skip-quantized", action="store_true")
    parser.add_argument("--skip-float16", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--no-trust-remote-code", action="store_false", dest="trust_remote_code"
    )
    parser.set_defaults(trust_remote_code=True)
    args = parser.parse_args()

    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"torch cuda device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        return 2

    selected = CASES
    if args.only:
        needle = args.only.lower()
        selected = [case for case in selected if needle in case.label.lower()]
    if args.skip_quantized:
        selected = [case for case in selected if not case.quantized]
    if args.skip_float16:
        selected = [case for case in selected if case.quantized]

    results: list[CheckResult] = []
    for index, case in enumerate(selected, start=1):
        source = case_source(case)
        print(f"\n[{index}/{len(selected)}] {case.label}")
        print(f"source: {source}", flush=True)
        ok, detail = run_case(case, args)
        results.append(CheckResult(case=case, source=source, passed=ok, detail=detail))
        if ok:
            print(f"PASS: {detail}", flush=True)
        else:
            print(f"FAIL:\n{detail}", flush=True)

    passed = [result for result in results if result.passed]
    failed = [result for result in results if not result.passed]
    print_result_table("Passed checks", passed)
    print_result_table("Failed checks", failed)
    print(
        f"\nSummary: {len(passed)} passed, {len(failed)} failed, {len(results)} total"
    )
    failures = len(failed)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
