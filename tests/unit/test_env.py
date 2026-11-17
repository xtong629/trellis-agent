import os
from pathlib import Path

import pytest

from trellis_agent.env import load_dotenv


def test_loads_key_value_lines_and_strips_quotes_and_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("T_A", "T_B", "T_C", "T_D"):
        monkeypatch.delenv(key, raising=False)
    file = tmp_path / ".env"
    file.write_text("# comment\nT_A=one\nT_B = \"two words\"\n\nnot a pair\nT_C='x'\nT_D=a=b\n")
    load_dotenv(file)
    assert (os.environ["T_A"], os.environ["T_B"], os.environ["T_C"]) == ("one", "two words", "x")
    assert os.environ["T_D"] == "a=b"  # only the first '=' splits


def test_never_overrides_existing_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("T_KEEP", "from-shell")
    file = tmp_path / ".env"
    file.write_text("T_KEEP=from-file\n")
    load_dotenv(file)
    assert os.environ["T_KEEP"] == "from-shell"


def test_missing_file_is_fine(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "nope.env")
