from types import SimpleNamespace
from unittest.mock import MagicMock

from one_dragon.base.push.push_service import PushService


def test_get_channel_config_normalizes_none_values() -> None:
    """推送配置未填写时应返回空字符串,避免渠道校验处理 None 崩溃。"""
    service = PushService(MagicMock())
    service._id_2_channel_schemas = {
        'SMTP': [
            SimpleNamespace(var_suffix='EMAIL', default=''),
            SimpleNamespace(var_suffix='PASSWORD', default=''),
        ]
    }
    push_config = MagicMock()
    push_config.get_channel_config_value.return_value = None
    service.__dict__['push_config'] = push_config

    assert service.get_channel_config('SMTP') == {'EMAIL': '', 'PASSWORD': ''}
