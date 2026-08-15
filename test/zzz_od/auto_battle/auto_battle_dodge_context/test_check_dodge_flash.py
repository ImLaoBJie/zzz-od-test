from concurrent.futures import Future
from unittest.mock import MagicMock

import numpy as np
import pytest

from zzz_od.auto_battle.auto_battle_dodge_context import (
    AudioTemplateEnum,
    AutoBattleDodgeContext,
    YoloStateEventEnum,
)


@pytest.mark.parametrize(
    ('audio_result', 'should_dodge'),
    [
        (AudioTemplateEnum.NORMAL_DODGE, True),
        (AudioTemplateEnum.PURPLE_DODGE, False),
        (AudioTemplateEnum.X_DODGE, True),
        (False, False),
    ],
)
def test_check_dodge_flash_only_accepts_templates_1_and_3(
    audio_result: AudioTemplateEnum | bool,
    should_dodge: bool,
) -> None:
    ctx = MagicMock()
    context = AutoBattleDodgeContext(ctx)
    context._flash_model = MagicMock()
    context._flash_model.run.return_value.class_idx = 0
    audio_future: Future[AudioTemplateEnum | bool] = Future()
    audio_future.set_result(audio_result)

    assert context.check_dodge_flash(np.empty((1, 1, 3)), 1.0, audio_future) is should_dodge
    if should_dodge:
        state_record = ctx.auto_battle_context.state_record_service.update_state.call_args.args[0]
        assert state_record.state_name == YoloStateEventEnum.DODGE_AUDIO.value
    else:
        ctx.auto_battle_context.state_record_service.update_state.assert_not_called()
