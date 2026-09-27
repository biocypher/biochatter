"""Check smoke-test safety guards without any network requests."""
# ruff: noqa: INP001
import sys
from types import SimpleNamespace

import pytest

from benchmark.medication_safety.scripts import live_smoke


def test_smoke_refuses_existing_directory_before_provider_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["smoke", "--output-dir", str(tmp_path)])

    def forbidden(**_kwargs):  # noqa: ANN003, ANN202
        pytest.fail("No provider should be created for an existing output directory")

    monkeypatch.setattr(live_smoke.runner, "create_conversation", forbidden)
    with pytest.raises(ValueError, match="fresh output directory"):
        live_smoke.main()


def test_cli_restores_arguments_after_failure(monkeypatch):
    original = ["test"]
    monkeypatch.setattr(sys, "argv", original)

    def fail():  # noqa: ANN202
        raise RuntimeError

    with pytest.raises(RuntimeError):
        live_smoke.invoke_cli(SimpleNamespace(__name__="dummy", main=fail), ["argument"])
    assert sys.argv is original


def test_smoke_rejects_invalid_token_budget_without_creating_outputs(tmp_path, monkeypatch):
    output = tmp_path / "fresh"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output-dir", str(output), "--generation-max-tokens", "0"])
    with pytest.raises(ValueError, match="token limit must be positive"):
        live_smoke.main()
    assert not output.exists()
