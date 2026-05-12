"""Tests del perfil LLM desde ``.env``."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config.llm_env import apply_copilot_llm_profile_from_dotenv


def test_llm_profile_off_forces_heuristic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "COPILOT_LLM_PROFILE=off\n"
        "COPILOT_QUERY_PLANNER=llm\n"
        "LLM_BASE_URL=https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COPILOT_QUERY_PLANNER", "llm")
    out = apply_copilot_llm_profile_from_dotenv(env)
    assert out == "off"
    assert os.environ["COPILOT_QUERY_PLANNER"] == "heuristic"


def test_llm_profile_openai_maps_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "COPILOT_LLM_PROFILE=openai\n"
        "OPENAI_LLM_BASE_URL=https://api.openai.com/v1\n"
        "OPENAI_LLM_MODEL=gpt-4o-mini\n"
        "OPENAI_API_KEY=sk-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    out = apply_copilot_llm_profile_from_dotenv(env)
    assert out == "openai"
    assert os.environ["LLM_BASE_URL"] == "https://api.openai.com/v1"
    assert os.environ["LLM_MODEL"] == "gpt-4o-mini"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"


def test_llm_profile_llamacpp_sets_dummy_key_if_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "COPILOT_LLM_PROFILE=llamacpp\n"
        "LLAMACPP_LLM_BASE_URL=http://127.0.0.1:8080/v1\n"
        "LLAMACPP_LLM_MODEL=local-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = apply_copilot_llm_profile_from_dotenv(env)
    assert out == "llamacpp"
    assert os.environ["LLM_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert os.environ["LLM_MODEL"] == "local-model"
    assert os.environ.get("OPENAI_API_KEY") == "sk-local-llamacpp"
