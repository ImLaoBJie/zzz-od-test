from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from zzz_od.auto_battle.auto_battle_dodge_context import (
    AudioTemplateEnum,
    AutoBattleDodgeContext,
)


@pytest.fixture
def dodge_context() -> AutoBattleDodgeContext:
    context = AutoBattleDodgeContext(MagicMock())
    context.init_audio_template()
    return context


def _get_audio_windows(context: AutoBattleDodgeContext) -> list[np.ndarray]:
    template_dir = Path('assets/template/dodge_audio')
    windows: list[np.ndarray] = []
    for template_path in sorted(template_dir.glob('template_*.wav')):
        wave = context._load_audio_template(template_path)
        if wave.size >= 16000:
            windows.append(wave[-16000:])
        else:
            windows.append(np.pad(wave, (16000 - wave.size, 0)))
    return windows


def test_check_dodge_audio_returns_template_number_or_false(
    dodge_context: AutoBattleDodgeContext,
) -> None:
    for template_idx, audio in enumerate(_get_audio_windows(dodge_context), start=1):
        dodge_context._audio_recorder.latest_audio = audio.copy()
        result = dodge_context.check_dodge_audio(float(template_idx))
        assert result == AudioTemplateEnum(template_idx)

    dodge_context._audio_recorder.latest_audio = np.zeros(16000)
    assert dodge_context.check_dodge_audio(4.0) is False


def test_check_dodge_audio_uses_highest_score(
    dodge_context: AutoBattleDodgeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dodge_context._audio_recorder.latest_audio = np.ones(16000)
    monkeypatch.setattr(
        dodge_context,
        'get_max_corr',
        lambda _: np.asarray([0.2, 0.8, 0.3]),
    )

    assert dodge_context.check_dodge_audio(1.0) == AudioTemplateEnum.PURPLE_DODGE

    dodge_context._audio_recorder.latest_audio = np.ones(16000)
    monkeypatch.setattr(
        dodge_context,
        'get_max_corr',
        lambda _: np.asarray([0.09, 0.08, 0.07]),
    )
    assert dodge_context.check_dodge_audio(2.0) is False
