"""Tests for stt_vocab.txt hotwords injection in _transcribe_local."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


def _fake_faster_whisper_module(mock_model):
    return SimpleNamespace(WhisperModel=MagicMock(return_value=mock_model))


def _make_mock_model():
    mock_segment = MagicMock()
    mock_segment.text = "hi"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 1.0
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    return mock_model


def _write_vocab(tmp_path: Path, contents: str) -> Path:
    home = tmp_path / "home"
    (home / ".hermes" / "state").mkdir(parents=True)
    vocab = home / ".hermes" / "state" / "stt_vocab.txt"
    vocab.write_text(contents, encoding="utf-8")
    return home


class TestLocalHotwords:

    def test_vocab_passed_as_hotwords(self, tmp_path):
        audio_file = tmp_path / "audio.ogg"
        audio_file.write_bytes(b"fake")
        home = _write_vocab(tmp_path, "RocketData\nHermes\nDenis")

        mock_model = _make_mock_model()
        fake_fw = _fake_faster_whisper_module(mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch.dict("sys.modules", {"faster_whisper": fake_fw}), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None), \
             patch("tools.transcription_tools.Path.home", return_value=home):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio_file), "base")

        assert result["success"] is True
        assert mock_model.transcribe.called
        _args, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("hotwords") == "RocketData Hermes Denis"

    def test_falls_back_to_initial_prompt_on_typeerror(self, tmp_path):
        audio_file = tmp_path / "audio.ogg"
        audio_file.write_bytes(b"fake")
        home = _write_vocab(tmp_path, "alpha beta")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock(language="en", duration=1.0)
        mock_model = MagicMock()

        def _transcribe_side_effect(path, **kwargs):
            if "hotwords" in kwargs:
                raise TypeError("unexpected keyword argument 'hotwords'")
            return ([mock_segment], mock_info)

        mock_model.transcribe.side_effect = _transcribe_side_effect
        fake_fw = _fake_faster_whisper_module(mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch.dict("sys.modules", {"faster_whisper": fake_fw}), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None), \
             patch("tools.transcription_tools.Path.home", return_value=home):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio_file), "base")

        assert result["success"] is True
        calls = mock_model.transcribe.call_args_list
        assert any("hotwords" in c.kwargs for c in calls)
        last_kwargs = calls[-1].kwargs
        assert last_kwargs.get("initial_prompt") == "alpha beta"
        assert "hotwords" not in last_kwargs

    def test_vocab_truncated_to_900_chars(self, tmp_path):
        audio_file = tmp_path / "audio.ogg"
        audio_file.write_bytes(b"fake")
        home = _write_vocab(tmp_path, "x" * 2000)

        mock_model = _make_mock_model()
        fake_fw = _fake_faster_whisper_module(mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch.dict("sys.modules", {"faster_whisper": fake_fw}), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None), \
             patch("tools.transcription_tools.Path.home", return_value=home):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio_file), "base")

        _args, kwargs = mock_model.transcribe.call_args
        assert len(kwargs["hotwords"]) == 900

    def test_no_vocab_file_no_hotwords_kwarg(self, tmp_path):
        audio_file = tmp_path / "audio.ogg"
        audio_file.write_bytes(b"fake")
        home = tmp_path / "empty-home"
        home.mkdir()

        mock_model = _make_mock_model()
        fake_fw = _fake_faster_whisper_module(mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch.dict("sys.modules", {"faster_whisper": fake_fw}), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None), \
             patch("tools.transcription_tools.Path.home", return_value=home):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio_file), "base")

        _args, kwargs = mock_model.transcribe.call_args
        assert "hotwords" not in kwargs
        assert "initial_prompt" not in kwargs
