"""Telegram must deliver one reply exactly once.

Covers the send-level dedup (identical text to the same chat within 60s is
not sent again, including after a streamed preview was finalized via edit)
and the inbound update_id dedup (a redelivered update is not processed twice).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    counter = iter(range(100, 1000))
    adapter._send_uncached = AsyncMock(
        side_effect=lambda *a, **k: SendResult(success=True, message_id=str(next(counter)))
    )
    adapter._edit_message_uncached = AsyncMock(
        side_effect=lambda chat_id, message_id, *a, **k: SendResult(success=True, message_id=message_id)
    )
    return adapter


@pytest.fixture
def clock(monkeypatch):
    from plugins.platforms.telegram import adapter as telegram_mod

    now = [1000.0]
    monkeypatch.setattr(telegram_mod.time, "monotonic", lambda: now[0])
    return now


# ── send-level dedup ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identical_text_sent_once_within_window(clock):
    adapter = _make_adapter()

    first = await adapter.send("123", "Готово, вот ответ")
    clock[0] += 30
    second = await adapter.send("123", "Готово, вот ответ")

    assert adapter._send_uncached.await_count == 1
    assert first.success and second.success
    assert second.message_id == first.message_id
    assert second.raw_response == {"deduplicated": True}


@pytest.mark.asyncio
async def test_whitespace_difference_still_deduplicated(clock):
    adapter = _make_adapter()

    await adapter.send("123", "same answer")
    await adapter.send("123", "same answer\n")

    assert adapter._send_uncached.await_count == 1


@pytest.mark.asyncio
async def test_identical_text_sent_again_after_window(clock):
    adapter = _make_adapter()

    await adapter.send("123", "same answer")
    clock[0] += 61
    await adapter.send("123", "same answer")

    assert adapter._send_uncached.await_count == 2


@pytest.mark.asyncio
async def test_different_chat_or_text_not_deduplicated(clock):
    adapter = _make_adapter()

    await adapter.send("123", "same answer")
    await adapter.send("456", "same answer")
    await adapter.send("123", "other answer")

    assert adapter._send_uncached.await_count == 3


@pytest.mark.asyncio
async def test_failed_send_is_not_remembered(clock):
    adapter = _make_adapter()
    adapter._send_uncached = AsyncMock(
        side_effect=[
            SendResult(success=False, error="boom"),
            SendResult(success=True, message_id="7"),
        ]
    )

    await adapter.send("123", "retry me")
    result = await adapter.send("123", "retry me")

    assert adapter._send_uncached.await_count == 2
    assert result.message_id == "7"


@pytest.mark.asyncio
async def test_final_send_after_streamed_finalize_edit_is_suppressed(clock):
    """Stream finalized the preview via edit; a later normal final send of the
    same text (fallback/broken stream path) must not create a duplicate."""
    adapter = _make_adapter()

    await adapter.edit_message("123", "55", "Полный ответ", finalize=True)
    result = await adapter.send("123", "Полный ответ", metadata={"notify": True})

    adapter._send_uncached.assert_not_awaited()
    assert result.success is True
    assert result.message_id == "55"


@pytest.mark.asyncio
async def test_non_final_edit_is_not_remembered(clock):
    adapter = _make_adapter()

    await adapter.edit_message("123", "55", "Полный ответ", finalize=False)
    await adapter.send("123", "Полный ответ")

    assert adapter._send_uncached.await_count == 1


@pytest.mark.asyncio
async def test_streaming_preview_send_bypasses_dedup(clock):
    """A preview that will be edited must get its own message, otherwise the
    stream would keep editing (and overwrite) the older reply."""
    adapter = _make_adapter()

    first = await adapter.send("123", "Ок")
    preview = await adapter.send("123", "Ок", metadata={"expect_edits": True})

    assert adapter._send_uncached.await_count == 2
    assert preview.message_id != first.message_id


@pytest.mark.asyncio
async def test_dedup_works_without_init_state(clock):
    """Adapters built via object.__new__ have no _recent_sent_texts yet."""
    adapter = _make_adapter()
    assert not hasattr(adapter, "_recent_sent_texts")

    await adapter.send("123", "hello")
    await adapter.send("123", "hello")

    assert adapter._send_uncached.await_count == 1


# ── inbound update dedup ─────────────────────────────────────────────


def test_redelivered_update_id_is_duplicate():
    adapter = _make_adapter()

    assert adapter._is_duplicate_update(SimpleNamespace(update_id=501)) is False
    assert adapter._is_duplicate_update(SimpleNamespace(update_id=501)) is True
    assert adapter._is_duplicate_update(SimpleNamespace(update_id=502)) is False


def test_update_without_int_id_is_never_duplicate():
    adapter = _make_adapter()

    assert adapter._is_duplicate_update(SimpleNamespace(update_id=None)) is False
    assert adapter._is_duplicate_update(SimpleNamespace(update_id=None)) is False


@pytest.mark.asyncio
async def test_handle_text_message_skips_redelivered_update():
    adapter = _make_adapter()
    adapter._should_process_message = lambda *a, **k: True
    adapter._ensure_forum_commands = AsyncMock()
    adapter._build_message_event = lambda *a, **k: SimpleNamespace(text="hi")
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._cache_replied_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    enqueued = []
    adapter._enqueue_text_event = enqueued.append

    msg = SimpleNamespace(text="hi")
    update = SimpleNamespace(update_id=900, message=msg, effective_message=msg)

    await adapter._handle_text_message(update, SimpleNamespace())
    await adapter._handle_text_message(update, SimpleNamespace())

    assert len(enqueued) == 1
