"""
tests/test_app_config.py
========================
Unit tests for ``gestureify.config.app_config.AppConfig``.

Covers:
  * Required field validation (missing SPOTIFY_CLIENT_ID).
  * Type coercion (string camera index → int).
  * Field-level validators (redirect URI scheme, log level).
  * Cross-field model validator (fist threshold < palm threshold).
  * Immutability (frozen model).
  * ``from_env()`` factory with mocked environment.

All tests are pure (no I/O, no side-effects).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import pydantic

from gestureify.config.app_config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ENV = {
    "SPOTIFY_CLIENT_ID": "abc123def456abc123def456abc12345",
    "SPOTIFY_REDIRECT_URI": "http://localhost:8080/callback",
    "CAMERA_INDEX": "0",
    "LOG_LEVEL": "INFO",
}


def _make_config(**overrides: object) -> AppConfig:
    """Return a valid AppConfig with optional field overrides."""
    data = {
        "spotify_client_id": "abc123def456abc123def456abc12345",
        "spotify_redirect_uri": "http://localhost:8080/callback",
    }
    data.update(overrides)
    return AppConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------


class TestValidConstruction:
    def test_minimal_valid_config(self) -> None:
        cfg = _make_config()
        assert cfg.spotify_client_id == "abc123def456abc123def456abc12345"

    def test_defaults_are_applied(self) -> None:
        cfg = _make_config()
        assert cfg.camera_index == 0
        assert cfg.log_level == "INFO"

    def test_log_level_is_uppercased(self) -> None:
        cfg = _make_config(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_frozen_model_rejects_mutation(self) -> None:
        cfg = _make_config()
        with pytest.raises((pydantic.ValidationError, TypeError)):
            cfg.camera_index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_missing_client_id_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            AppConfig.model_validate(
                {"spotify_redirect_uri": "http://localhost:8080/callback"}
            )
        errors = exc_info.value.errors()
        locs = [e["loc"] for e in errors]
        assert ("spotify_client_id",) in locs

    def test_empty_client_id_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _make_config(spotify_client_id="")


# ---------------------------------------------------------------------------
# Field-level validators
# ---------------------------------------------------------------------------


class TestFieldValidators:
    def test_redirect_uri_must_start_with_http(self) -> None:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            _make_config(spotify_redirect_uri="ftp://localhost:8080/callback")
        assert any(
            "http" in str(e["msg"]).lower() for e in exc_info.value.errors()
        )

    def test_https_redirect_uri_is_valid(self) -> None:
        cfg = _make_config(spotify_redirect_uri="https://myapp.example.com/callback")
        assert cfg.spotify_redirect_uri.startswith("https://")

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _make_config(log_level="VERBOSE")

    def test_camera_index_out_of_range_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _make_config(camera_index=999)

    def test_camera_index_negative_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _make_config(camera_index=-1)


# ---------------------------------------------------------------------------
# Cross-field model validator
# ---------------------------------------------------------------------------


class TestModelValidator:
    def test_fist_threshold_above_palm_threshold_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            _make_config(
                fist_ratio_threshold=3.0,
                wake_palm_ratio_threshold=1.0,
            )
        assert any(
            "fist_ratio_threshold" in str(e["msg"])
            for e in exc_info.value.errors()
        )

    def test_fist_equal_to_palm_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _make_config(
                fist_ratio_threshold=1.5,
                wake_palm_ratio_threshold=1.5,
            )

    def test_valid_threshold_ordering_passes(self) -> None:
        cfg = _make_config(
            fist_ratio_threshold=0.6,
            wake_palm_ratio_threshold=1.8,
        )
        assert cfg.fist_ratio_threshold < cfg.wake_palm_ratio_threshold


# ---------------------------------------------------------------------------
# from_env() factory
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_from_env_reads_client_id(self) -> None:
        with patch.dict(os.environ, _VALID_ENV, clear=False):
            cfg = AppConfig.from_env()
        assert cfg.spotify_client_id == _VALID_ENV["SPOTIFY_CLIENT_ID"]

    def test_from_env_missing_client_id_raises(self) -> None:
        env = {k: v for k, v in _VALID_ENV.items() if k != "SPOTIFY_CLIENT_ID"}
        with patch.dict(os.environ, env, clear=False):
            # Ensure the key is absent even if it exists in the real environment.
            env_without = {**os.environ, **env}
            env_without.pop("SPOTIFY_CLIENT_ID", None)
            with patch.dict(os.environ, {"SPOTIFY_CLIENT_ID": ""}, clear=False):
                with pytest.raises(pydantic.ValidationError):
                    AppConfig.from_env()

    def test_from_env_camera_index_coerced_from_string(self) -> None:
        with patch.dict(os.environ, {**_VALID_ENV, "CAMERA_INDEX": "2"}, clear=False):
            cfg = AppConfig.from_env()
        assert cfg.camera_index == 2
