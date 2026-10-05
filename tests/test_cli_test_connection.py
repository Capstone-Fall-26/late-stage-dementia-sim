from __future__ import annotations

from typer.testing import CliRunner

from cli.app import app

runner = CliRunner()


def test_test_connection_mock_backend_succeeds() -> None:
    result = runner.invoke(app, ["test-connection", "--backend", "mock"])
    assert result.exit_code == 0
    assert "Connected" in result.stdout


def test_test_connection_unknown_backend_fails() -> None:
    result = runner.invoke(app, ["test-connection", "--backend", "not-a-backend"])
    assert result.exit_code != 0
