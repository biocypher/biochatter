"""Opt-in paid DeepSeek integration check; never run automatically in CI."""
# Wrap dynamic provider methods; fixed counts are test assertions.
# ruff: noqa: ANN002, ANN003, ANN202, PLR2004, SLF001, T201

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark.medication_safety.scripts import judge_with_biochatter as judge
from benchmark.medication_safety.scripts import run_with_biochatter as runner
from benchmark.medication_safety.scripts import score_responses as scorer


def invoke_cli(module, arguments):
    """Exercise the actual command-line entry point."""
    previous = sys.argv
    try:
        sys.argv = [module.__name__, *map(str, arguments)]
        module.main()
    finally:
        sys.argv = previous


def main():  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--generation-max-tokens", type=int, default=8192)
    parser.add_argument("--system-prompt", choices=["none", "minimal"], default="none")
    parser.add_argument(
        "--patient-attitude", choices=["neutral", "confident", "anxious", "very_anxious"], default="neutral"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.generation_max_tokens < 1:
        message = "Generation token limit must be positive."
        raise ValueError(message)
    directory = args.output_dir
    if directory.exists():
        message = "Use a fresh output directory for an auditable smoke test."
        raise ValueError(message)
    directory.mkdir(parents=True)
    counts = {"generation": 0, "judge": 0}
    observed_models = set()
    request_metadata = []
    original_factory = runner.create_conversation

    def factory(kind):
        def create(**kwargs):
            conversation = original_factory(**kwargs)
            # Bound time and cost: do not let the SDK silently repeat requests.
            conversation.chat._chat.max_retries = 0
            conversation.chat._chat.request_timeout = 120
            conversation.chat._chat.root_client.max_retries = 0
            conversation.chat._chat.root_client.timeout = 120
            original_generate = conversation.chat.generate

            def generate(*a, **kw):
                response = original_generate(*a, **kw)
                (directory / f"{kind}_{counts[kind]}_output.txt").write_text(
                    response.generations[0][0].text,
                    encoding="utf-8",
                )
                metadata = response.llm_output or {}
                if metadata.get("model_name"):
                    observed_models.add(metadata["model_name"])
                request_metadata.append(
                    {
                        "stage": kind,
                        "model": metadata.get("model_name"),
                        "usage": metadata.get("token_usage"),
                        "finish_reasons": [
                            g.generation_info.get("finish_reason") if g.generation_info else None
                            for group in response.generations
                            for g in group
                        ],
                    }
                )
                (directory / "request_metadata.json").write_text(
                    json.dumps(request_metadata, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"Completed {kind} request {counts[kind]}", flush=True)
                if any(
                    g.generation_info and g.generation_info.get("finish_reason") == "length"
                    for group in response.generations
                    for g in group
                ):
                    message = "Provider truncated the response; smoke test failed."
                    raise RuntimeError(message)
                return response

            conversation.chat.generate = generate
            query = conversation.query

            def counted(prompt):
                counts[kind] += 1
                if counts[kind] > (4 if kind == "generation" else 24):
                    message = "Smoke-test request budget exceeded."
                    raise RuntimeError(message)
                result = query(prompt)
                if not result[0] or result[1] is None:
                    message = "Provider returned no usable response or usage."
                    raise RuntimeError(message)
                return result

            conversation.query = counted
            return conversation

        return create

    runner.create_conversation = factory("generation")
    judge.create_conversation = factory("judge")
    responses = directory / "responses.csv"
    judgements = directory / "judgements.csv"
    scores = directory / "communication.csv"
    common = ["--provider", "deepseek", "--model", args.model, "--api-key-env", args.api_key_env, "--temperature", "0"]
    generation = [
        *common,
        "--max-tokens",
        str(args.generation_max_tokens),
        "--thinking",
        "disabled",
        "--iterations",
        "4",
        "--limit",
        "1",
        "--system-prompt",
        args.system_prompt,
        "--patient-attitude",
        args.patient_attitude,
        "--output",
        responses,
    ]
    judging = [
        responses,
        *common,
        "--max-tokens",
        "4096",
        "--thinking",
        "enabled",
        "--reasoning-effort",
        "high",
        "--judge-iterations",
        "2",
        "--max-attempts",
        "1",
        "--output",
        judgements,
        "--scores-output",
        scores,
    ]
    try:
        print("Generating four responses through BioChatter", flush=True)
        invoke_cli(runner, generation)
        if len(judge.load_response_rows(responses)) != 4:
            message = "Expected four generated responses."
            raise RuntimeError(message)
        invoke_cli(scorer, [responses, "--output", directory / "structured.csv"])
        print("Running 24 criterion-level judgements through BioChatter", flush=True)
        invoke_cli(judge, judging)
        raw = pd.read_csv(judgements)
        summary = pd.read_csv(scores)
        if len(raw) != 24 or len(summary) != 12 or not summary.judge_iterations.eq(2).all():
            message = "Unexpected judgement or score dimensions."
            raise RuntimeError(message)
        before = counts.copy()
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [responses, judgements, scores]}
        print("Checking resume without additional generation/judgement requests", flush=True)
        invoke_cli(runner, generation)
        invoke_cli(judge, judging)
        if counts != before or any(
            hashlib.sha256((directory / n).read_bytes()).hexdigest() != h for n, h in hashes.items()
        ):
            message = "Resume changed outputs or issued extra inference."
            raise RuntimeError(message)
        report = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "requested_model": args.model,
            "system_prompt": args.system_prompt,
            "patient_attitude": args.patient_attitude,
            "provider_reported_models": sorted(observed_models),
            "inference_calls": counts,
            "structured_rows": len(pd.read_csv(directory / "structured.csv")),
            "criterion_scores": len(summary),
            "resume_unchanged": True,
            "generation_settings": {"temperature": 0, "max_tokens": args.generation_max_tokens, "thinking": "disabled"},
            "judge_settings": {"temperature": 0, "max_tokens": 4096, "thinking": "enabled", "reasoning_effort": "high"},
            "scope": "One instance; pipeline check, not a replication of study results or all providers.",
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        runner.create_conversation = original_factory
        judge.create_conversation = original_factory


if __name__ == "__main__":
    main()
