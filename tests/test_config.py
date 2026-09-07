"""Settings: defaults, env overrides, derived paths, directory creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_rag.config import Settings, settings

_ENV_KEYS = (
    "LLM_BASE_URL",
    "LLM_MODEL",
    "EMBED_BASE_URL",
    "EMBED_MODEL",
    "VLM_BASE_URL",
    "VLM_MODEL",
    "OPENAI_API_KEY",
    "COLPALI_MODEL",
    "COLPALI_DEVICE",
    "DATA_DIR",
    "STORAGE_DIR",
    "TEXT_TOP_K",
    "VISUAL_TOP_K",
    "MAX_AGENT_ITERATIONS",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_point_at_the_llama_cpp_ports(clean_env: None) -> None:
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://127.0.0.1:8080/v1"
    assert s.embed_base_url == "http://127.0.0.1:8081/v1"
    assert s.vlm_base_url == "http://127.0.0.1:8082/v1"
    assert s.openai_api_key == "sk-no-key-required"
    assert s.colpali_model == "vidore/colqwen2-v1.0"
    assert s.colpali_device == "mps"
    assert (s.text_top_k, s.visual_top_k, s.max_agent_iterations) == (5, 3, 3)
    assert s.data_dir == Path("./data")
    assert s.storage_dir == Path("./storage")


def test_env_vars_override_defaults(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "ministral-3-14b")
    monkeypatch.setenv("TEXT_TOP_K", "9")
    monkeypatch.setenv("MAX_AGENT_ITERATIONS", "1")
    monkeypatch.setenv("STORAGE_DIR", "/srv/rag-storage")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://127.0.0.1:1234/v1"
    assert s.llm_model == "ministral-3-14b"
    assert s.text_top_k == 9
    assert s.max_agent_iterations == 1
    assert s.storage_dir == Path("/srv/rag-storage")


def test_env_var_names_are_case_insensitive(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("colpali_device", "cpu")
    assert Settings(_env_file=None).colpali_device == "cpu"


def test_non_integer_top_k_is_rejected(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TOP_K", "five")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_unknown_env_vars_are_ignored(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_UNRELATED_SETTING", "1")
    Settings(_env_file=None)  # must not raise thanks to extra="ignore"


def test_derived_storage_paths_hang_off_storage_dir(clean_env: None) -> None:
    s = Settings(_env_file=None, storage_dir=Path("/srv/rag"))
    assert s.chroma_dir == Path("/srv/rag/chroma")
    assert s.colpali_dir == Path("/srv/rag/colpali")
    assert s.page_image_dir == Path("/srv/rag/page_images")


def test_ensure_dirs_creates_every_path_and_is_idempotent(clean_env: None, tmp_path: Path) -> None:
    s = Settings(_env_file=None, storage_dir=tmp_path / "storage", data_dir=tmp_path / "data")
    s.ensure_dirs()
    for d in (s.data_dir, s.storage_dir, s.chroma_dir, s.colpali_dir, s.page_image_dir):
        assert d.is_dir(), d
    s.ensure_dirs()  # second call must not raise on existing dirs


def test_module_level_singleton_is_a_settings_instance() -> None:
    assert isinstance(settings, Settings)
