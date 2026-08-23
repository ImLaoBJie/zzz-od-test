"""MCP 2.0 公共客户端接口的回归测试。"""

import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
import uvicorn
from mcp.client import Client
from mcp.server import MCPServer

from zzz_od.backend.mcp.app import create_mcp_server
from zzz_od.backend.schemas import RunStatusResult, WindowStatus


def _make_server() -> tuple[MCPServer, MagicMock]:
    """构造不依赖真实游戏的 MCPServer 与伪造 backend。"""
    backend = MagicMock()
    backend.check_window.return_value = WindowStatus(
        win_title='ZenlessZoneZero',
        is_win_valid=True,
        is_win_active=False,
        is_win_scale=True,
        x=10,
        y=20,
        width=1920,
        height=1080,
    )
    return create_mcp_server(backend), backend


def test_create_mcp_server_returns_mcp_v2_server() -> None:
    """服务工厂应直接返回 MCP 2.0 的 MCPServer。"""
    server, _ = _make_server()
    assert isinstance(server, MCPServer)


@pytest.mark.asyncio
async def test_mcp_v2_client_lists_tools_and_prompts() -> None:
    """MCP 2.0 客户端应能完成握手并读取 tool、prompt 清单。"""
    server, _ = _make_server()

    async with Client(server) as client:
        tools = await client.list_tools()
        prompts = await client.list_prompts()
        prompt = await client.get_prompt('zzz_check_status')

    tool_names = {tool.name for tool in tools.tools}
    prompt_names = {prompt.name for prompt in prompts.prompts}
    assert {'check_game_window', 'analyze_screen', 'open_game'} <= tool_names
    assert {'zzz_check_status', 'zzz_run_one_dragon'} <= prompt_names
    assert any('get_run_status' in message.content.text for message in prompt.messages)


@pytest.mark.asyncio
async def test_mcp_v2_client_calls_sync_tool_with_structured_result() -> None:
    """MCP 2.0 客户端调用同步 tool 时应保留结构化返回与 backend 委托。"""
    server, backend = _make_server()

    async with Client(server) as client:
        result = await client.call_tool('check_game_window')

    assert result.is_error is False
    assert result.structured_content == {
        'result': {
            'win_title': 'ZenlessZoneZero',
            'is_win_valid': True,
            'is_win_active': False,
            'is_win_scale': True,
            'x': 10,
            'y': 20,
            'width': 1920,
            'height': 1080,
        },
    }
    backend.check_window.assert_called_once_with()


@pytest.mark.asyncio
async def test_mcp_v2_client_calls_async_tool() -> None:
    """MCP 2.0 客户端调用异步 tool 时应保留非阻塞运行返回。"""
    server, backend = _make_server()
    backend.start_run.return_value = (True, Future())
    backend.query_status.return_value = RunStatusResult(
        state='running',
        source='mcp',
        app='OpenGame',
        started_at='2026-08-23T00:00:00',
        duration_seconds=0.0,
    )

    async with Client(server) as client:
        result = await client.call_tool(
            'open_game',
            {'enter': False, 'block': False},
        )

    assert result.is_error is False
    assert result.structured_content['result']['started'] is True
    assert result.structured_content['result']['source'] == 'mcp'
    backend.start_run.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_v2_streamable_http_transport(unused_tcp_port: int) -> None:
    """真实本机 HTTP 传输应能握手、列举工具并调用同步 tool。"""
    server, backend = _make_server()
    app = server.streamable_http_app()
    uvicorn_server = uvicorn.Server(uvicorn.Config(
        app,
        host='127.0.0.1',
        port=unused_tcp_port,
        log_level='warning',
        access_log=False,
    ))
    serve_task = asyncio.create_task(uvicorn_server.serve())
    try:
        for _ in range(100):
            if uvicorn_server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError('测试用 MCP HTTP 服务未能启动')

        async with Client(f'http://127.0.0.1:{unused_tcp_port}/mcp') as client:
            tools = await client.list_tools()
            result = await client.call_tool('check_game_window')

        assert any(tool.name == 'check_game_window' for tool in tools.tools)
        assert result.is_error is False
        assert result.structured_content['result']['is_win_valid'] is True
        backend.check_window.assert_called_once_with()
    finally:
        uvicorn_server.should_exit = True
        await serve_task
