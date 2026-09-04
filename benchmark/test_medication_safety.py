"""Generate medication safety responses through BioChatter."""

import os

import pytest

from .medication_safety.biochatter_adapter import (
    default_response_path,
    response_already_recorded,
    run_and_record_instance,
)


@pytest.mark.medication_safety
def test_generate_medication_safety_responses(
    model_name,
    conversation,
    test_data_medication_safety,
):
    """Run one expanded medication safety instance for one configured model."""
    iterations = int(os.getenv("BIOCHATTER_MEDICATION_SAFETY_ITERATIONS", "4"))
    output_path = default_response_path(model_name)

    if response_already_recorded(
        output_path,
        model_name,
        test_data_medication_safety["hash"],
    ):
        pytest.skip("Medication safety response already recorded.")

    run_and_record_instance(
        conversation=conversation,
        model_name=model_name,
        instance=test_data_medication_safety,
        output_path=output_path,
        iterations=iterations,
    )
