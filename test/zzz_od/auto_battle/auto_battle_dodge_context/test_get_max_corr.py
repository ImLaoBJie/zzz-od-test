from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from scipy.signal import correlate

from zzz_od.auto_battle.auto_battle_dodge_context import AutoBattleDodgeContext


def _legacy_get_max_corr(
    context: AutoBattleDodgeContext,
    template: np.ndarray,
    audio: np.ndarray,
) -> float:
    """保留旧实现的逐模板算法，作为新批量算法的测试基线。"""
    filtered_audio = context._get_filter_wave(audio)
    scaled_template = template / np.std(template)
    scaled_audio = filtered_audio / np.std(filtered_audio)
    if scaled_template.size > scaled_audio.size:
        correlation = correlate(
            scaled_template,
            scaled_audio,
            mode='same',
            method='fft',
        ) / scaled_template.size
    else:
        correlation = correlate(
            scaled_audio,
            scaled_template,
            mode='same',
            method='fft',
        ) / scaled_audio.size
    return float(np.max(correlation))


def test_get_max_corr_matches_legacy_algorithm() -> None:
    context = AutoBattleDodgeContext(MagicMock())
    context.init_audio_template()
    template_paths = sorted(Path('assets/template/dodge_audio').glob('template_*.wav'))
    waves = [context._load_audio_template(path) for path in template_paths]
    templates = [context._get_filter_wave(wave) for wave in waves]

    for expected_idx, wave in enumerate(waves):
        audio = wave[-16000:] if wave.size >= 16000 else np.pad(wave, (16000 - wave.size, 0))
        legacy_scores = np.asarray([
            _legacy_get_max_corr(context, template, audio)
            for template in templates
        ])
        batch_scores = context.get_max_corr(audio)

        np.testing.assert_allclose(batch_scores, legacy_scores, atol=1e-12)
        assert int(np.argmax(batch_scores)) == expected_idx
        assert batch_scores[expected_idx] > 0.85
        assert np.max(np.delete(batch_scores, expected_idx)) < 0.1
