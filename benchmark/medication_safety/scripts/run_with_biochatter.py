"""Run the medication safety benchmark through BioChatter providers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmark.medication_safety.biochatter_adapter import (
    DEFAULT_ITERATIONS,
    default_response_path,
    load_medication_safety_instances,
    run_and_record_instance,
)
from benchmark.medication_safety.biochatter_provider import (
    DEFAULT_KEY_ENV,
    create_conversation,
)


logger = logging.getLogger(__name__)


def select_instances(instances, args):
    """Apply optional command-line filters to expanded instances."""
    selected = [
        instance
        for instance in instances
        if (not args.case_id or instance["case_id"] in args.case_id)
        and (not args.system_prompt or instance["system_prompt"] in args.system_prompt)
        and (not args.patient_attitude or instance["patient_attitude"] in args.patient_attitude)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate medication safety responses through BioChatter.",
    )
    parser.add_argument("--provider", choices=sorted(DEFAULT_KEY_ENV), required=True)
    parser.add_argument("--model", required=True, help="Provider model identifier.")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key.")
    parser.add_argument("--base-url", help="Base URL for an OpenAI-compatible provider.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--system-prompt", action="append")
    parser.add_argument("--patient-attitude", action="append")
    parser.add_argument("--limit", type=int, help="Run only the first N selected instances.")
    parser.add_argument("--output", type=Path, help="Local response CSV path.")
    parser.add_argument("--force", action="store_true", help="Replace matching local records.")
    args = parser.parse_args()

    conversation = create_conversation(
        provider=args.provider,
        model_name=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
    )
    instances = select_instances(load_medication_safety_instances(), args)
    if not instances:
        msg = "No medication safety instances matched the selected filters."
        raise RuntimeError(msg)

    output_path = args.output or default_response_path(args.model)
    for index, instance in enumerate(instances, start=1):
        run_and_record_instance(
            conversation=conversation,
            model_name=args.model,
            instance=instance,
            output_path=output_path,
            iterations=args.iterations,
            force=args.force,
        )
        logger.info("[%s/%s] %s", index, len(instances), instance["case"])

    logger.info("Responses written to %s", output_path)


if __name__ == "__main__":
    main()
