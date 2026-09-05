"""A voice note queued while the agent is busy is transcribed and echoed ONCE.

Regression for the double 🎙️ echo / double STT run seen in Telegram DMs with
``busy_input_mode: queue``.  The same cached audio file went through local
whisper (~30s CPU each) and was echoed as ``🎙️ "…"`` twice:

  1. the post-run pending drain in ``GatewayRunner._run_agent`` transcribed +
     echoed the dequeued event, then
  2. ``_prepare_inbound_message_text`` transcribed + echoed it AGAIN when the
     event became the next user turn (three times when the interrupt monitor
     had also peeked at the queued event first).

The enriched text is now cached on the ``MessageEvent``
(``_gateway_pending_stt_text`` / ``_gateway_pending_stt_transcripts``) with an
echo ledger (``_gateway_pending_stt_echoed``), and every consumer reuses it.
No network, no whisper: ``transcribe_audio`` is patched throughout.
"""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    merge_pending_message_event,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource

SESSION_KEY = "agent:main:telegram:dm:12345"
TRANSCRIBE = "tools.transcription_tools.transcribe_audio"


def _ok(transcript: str) -> dict:
    return {"success": True, "transcript": transcript, "provider": "mock"}


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")


def _voice_event(path: str, text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=[path],
        media_types=["audio/ogg"],
    )


def _runner(adapter=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        stt_enabled=True,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._consume_pending_native_image_paths = lambda _key: []
    runner._session_key_for_source = lambda _source: SESSION_KEY
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
    runner._has_setup_skill = lambda: False
    return runner


def _echoes(adapter) -> list:
    return [content for (_chat, content, _meta) in adapter.sent if content.startswith("🎙️")]


class _CaptureAdapter(BasePlatformAdapter):
    """Records every outbound send so the tests can count 🎙️ echoes."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


# ---------------------------------------------------------------------------
# Helper level: peek (interrupt monitor) + drain share one transcript/echo
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_peek_then_drain_transcribes_and_echoes_once():
    adapter = SimpleNamespace(send=AsyncMock())
    runner = _runner(adapter)
    source = _source()
    event = _voice_event("/tmp/audio_4b2c7b471fb6.ogg")

    with patch(TRANSCRIBE, return_value=_ok("hello once")) as mock_transcribe:
        peek_text, peek_tx = await runner._transcribe_and_echo_pending_voice(
            event, adapter, source, event.text, log_context="Voice-interrupt",
        )
        drain_text, drain_tx = await runner._transcribe_and_echo_pending_voice(
            event, adapter, source, event.text, log_context="Voice-drain",
        )

    assert "hello once" in peek_text
    assert drain_text == peek_text
    assert peek_tx == drain_tx == ["hello once"]
    mock_transcribe.assert_called_once_with("/tmp/audio_4b2c7b471fb6.ogg")
    adapter.send.assert_awaited_once_with("12345", '🎙️ "hello once"', metadata=None)
    # Cache + ledger live on the event, not on the runner.
    assert event._gateway_pending_stt_text == peek_text
    assert event._gateway_pending_stt_transcripts == ["hello once"]
    assert event._gateway_pending_stt_echoed == 1


@pytest.mark.asyncio
async def test_prepare_inbound_reuses_drain_transcript_without_second_stt():
    """The bug: drain transcribed, then _prepare_inbound_message_text did it again."""
    adapter = _CaptureAdapter()
    runner = _runner(adapter)
    source = _source()
    event = _voice_event("/tmp/audio_4b2c7b471fb6.ogg")

    with patch(TRANSCRIBE, return_value=_ok("hello once")) as mock_transcribe:
        drained, _ = await runner._transcribe_and_echo_pending_voice(
            event, adapter, source, event.text, log_context="Voice-drain",
        )
        prepared = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

    mock_transcribe.assert_called_once_with("/tmp/audio_4b2c7b471fb6.ogg")
    assert _echoes(adapter) == ['🎙️ "hello once"']
    assert prepared is not None
    assert "hello once" in prepared
    assert "voice message" in prepared.lower()
    # The agent sees the same enriched wrapper the drain produced.
    assert prepared == drained


@pytest.mark.asyncio
async def test_fresh_voice_still_transcribed_by_prepare_inbound():
    """No pending cache on the event -> the normal inbound STT path is untouched."""
    adapter = _CaptureAdapter()
    runner = _runner(adapter)
    event = _voice_event("/tmp/audio_fresh.ogg")

    with patch(TRANSCRIBE, return_value=_ok("fresh note")) as mock_transcribe:
        prepared = await runner._prepare_inbound_message_text(
            event=event, source=_source(), history=[],
        )

    mock_transcribe.assert_called_once_with("/tmp/audio_fresh.ogg")
    assert "fresh note" in prepared
    assert _echoes(adapter) == ['🎙️ "fresh note"']


@pytest.mark.asyncio
async def test_audio_file_attachment_is_not_transcribed_while_queued():
    """MessageType.AUDIO (file attachment) must never enter STT — same rule as inbound (#24870)."""
    adapter = _CaptureAdapter()
    runner = _runner(adapter)
    event = MessageEvent(
        text="",
        message_type=MessageType.AUDIO,
        source=_source(),
        media_urls=["/tmp/song.mp3"],
        media_types=["audio/mpeg"],
    )

    with patch(TRANSCRIBE, side_effect=AssertionError("must not STT audio attachments")):
        text, transcripts = await runner._transcribe_and_echo_pending_voice(
            event, adapter, _source(), event.text, log_context="Voice-drain",
        )

    assert (text, transcripts) == ("", [])
    assert adapter.sent == []
    assert not hasattr(event, "_gateway_pending_stt_text")


# ---------------------------------------------------------------------------
# Media merge: a second voice note merged into the queued event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_merged_second_note_transcribes_only_new_clip_and_echoes_each_once():
    adapter = _CaptureAdapter()
    runner = _runner(adapter)
    source = _source()
    pending = {}
    first = _voice_event("/tmp/audio_first.ogg")
    merge_pending_message_event(pending, SESSION_KEY, first)

    transcripts = {"/tmp/audio_first.ogg": "first note", "/tmp/audio_second.ogg": "second note"}

    with patch(TRANSCRIBE, side_effect=lambda p: _ok(transcripts[p])) as mock_transcribe:
        # Interrupt monitor peeks and transcribes the first note.
        await runner._transcribe_and_echo_pending_voice(
            pending[SESSION_KEY], adapter, source, "", log_context="Voice-interrupt",
        )
        # A second voice note arrives while still busy and merges into the slot.
        merge_pending_message_event(pending, SESSION_KEY, _voice_event("/tmp/audio_second.ogg"))
        merged = pending[SESSION_KEY]
        assert merged is first
        assert merged.media_urls == ["/tmp/audio_first.ogg", "/tmp/audio_second.ogg"]
        # Derived cache dropped, echo ledger kept.
        assert not hasattr(merged, "_gateway_pending_stt_text")
        assert merged._gateway_pending_stt_echoed == 1

        # Drain re-enriches: first clip comes from the memo, second runs STT.
        drained, tx = await runner._transcribe_and_echo_pending_voice(
            merged, adapter, source, "", log_context="Voice-drain",
        )
        prepared = await runner._prepare_inbound_message_text(
            event=merged, source=source, history=[],
        )

    assert [c.args[0] for c in mock_transcribe.call_args_list] == [
        "/tmp/audio_first.ogg",
        "/tmp/audio_second.ogg",
    ]
    assert tx == ["first note", "second note"]
    assert _echoes(adapter) == ['🎙️ "first note"', '🎙️ "second note"']
    assert "first note" in drained and "second note" in drained
    assert prepared == drained


# ---------------------------------------------------------------------------
# Path-keyed transcript memo in _enrich_message_with_transcription
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcript_memo_is_bounded_and_skips_failures():
    runner = _runner()
    runner._STT_TRANSCRIPT_MEMO_MAX = 2

    with patch(TRANSCRIBE, side_effect=lambda p: _ok(f"tx:{p}")) as mock_transcribe:
        for path in ("/tmp/a.ogg", "/tmp/b.ogg", "/tmp/c.ogg"):
            await runner._enrich_message_with_transcription("", [path])
        # "a" was evicted (bounded to 2) -> transcribed again; "c" is a hit.
        await runner._enrich_message_with_transcription("", ["/tmp/a.ogg"])
        _, tx = await runner._enrich_message_with_transcription("", ["/tmp/c.ogg"])

    assert tx == ["tx:/tmp/c.ogg"]
    assert [c.args[0] for c in mock_transcribe.call_args_list] == [
        "/tmp/a.ogg", "/tmp/b.ogg", "/tmp/c.ogg", "/tmp/a.ogg",
    ]
    assert len(runner._stt_transcript_memo()) <= 2

    # Failures are not memoised: the next call retries STT.
    with patch(TRANSCRIBE, return_value={"success": False, "error": "boom"}) as mock_fail:
        await runner._enrich_message_with_transcription("", ["/tmp/z.ogg"])
        await runner._enrich_message_with_transcription("", ["/tmp/z.ogg"])
    assert mock_fail.call_count == 2


# ---------------------------------------------------------------------------
# End to end through _run_agent: queue mode (no interrupt), drain -> next turn
# ---------------------------------------------------------------------------

class _TwoTurnAgent:
    """First turn finishes normally; the queued voice note becomes turn two."""

    messages: list = []

    def __init__(self, **kwargs):
        self.tools = []
        self.model = "test-model"
        self.provider = "test-provider"
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def interrupt(self, message=None):
        self._interrupt_requested = True

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).messages.append(message)
        if len(type(self).messages) == 1:
            return {"final_response": "first turn done", "messages": [], "api_calls": 1}
        return {"final_response": "follow-up complete", "messages": [], "api_calls": 1}


def _run_agent_runner(adapter) -> GatewayRunner:
    runner = _runner(adapter)
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._queued_events = {}
    runner._draining = False
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=True,
    )
    return runner


@pytest.mark.asyncio
async def test_run_agent_drains_queued_voice_with_single_stt_and_single_echo(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _TwoTurnAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    adapter = _CaptureAdapter()
    runner = _run_agent_runner(adapter)
    source = _source()
    voice_path = str(tmp_path / "audio_4b2c7b471fb6.ogg")
    event = _voice_event(voice_path)

    # busy_input_mode: queue — the note sits in the slot WITHOUT an interrupt.
    adapter._pending_messages[SESSION_KEY] = event
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    _TwoTurnAgent.messages = []

    with patch(TRANSCRIBE, return_value=_ok("hello once")) as mock_transcribe:
        result = await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-voice-once",
            session_key=SESSION_KEY,
        )

    assert result["final_response"] == "follow-up complete"
    assert len(_TwoTurnAgent.messages) == 2
    assert _TwoTurnAgent.messages[0] == "hi"
    assert "hello once" in _TwoTurnAgent.messages[1]
    mock_transcribe.assert_called_once_with(voice_path)
    assert _echoes(adapter) == ['🎙️ "hello once"']
    assert SESSION_KEY not in adapter._pending_messages
