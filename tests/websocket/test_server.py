import asyncio
import pytest
from app.websocket import server
from app.websocket.server import _handler, _ws_server_task, run_ws_server, broadcast


@pytest.fixture(autouse=True)
def clean_server_state():
    """Garante que o estado global do servidor WebSocket seja limpo antes e depois de cada teste."""
    server._clients.clear()
    server._loop = None
    yield
    server._clients.clear()
    server._loop = None


@pytest.mark.asyncio
async def test_handler_ping_pong_e_desconexao(mocker):
    mock_ws = mocker.AsyncMock()
    mock_ws.__aiter__.return_value = ["ping"]

    await _handler(mock_ws)

    mock_ws.send.assert_called_once_with("pong")
    assert mock_ws not in server._clients


@pytest.mark.asyncio
async def test_handler_trata_excecao(mocker):
    mock_ws = mocker.AsyncMock()
    # Força uma exceção durante a iteração das mensagens para cobrir o bloco except
    mock_ws.__aiter__.side_effect = Exception("Conexão interrompida")

    await _handler(mock_ws)

    # Garante que o cliente foi removido no bloco finally após o erro
    assert mock_ws not in server._clients


@pytest.mark.asyncio
async def test_ws_server_task(mocker):
    mock_serve = mocker.patch("websockets.serve", return_value=mocker.AsyncMock())
    mocker.patch("asyncio.Future", side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await _ws_server_task()

    assert server._loop is not None
    mock_serve.assert_called_once()


def test_run_ws_server(mocker):
    mock_run = mocker.patch("asyncio.run")
    run_ws_server()
    mock_run.assert_called_once()


def test_broadcast_sem_clientes_ou_sem_loop():
    server._clients = set()
    server._loop = None

    # Executa broadcast sem nenhum cliente ou loop configurado
    broadcast({"teste": "data"})


def test_broadcast_com_clientes_conectados(mocker):
    mock_ws = mocker.AsyncMock()
    mock_loop = mocker.MagicMock()

    server._clients = {mock_ws}
    server._loop = mock_loop

    payload = {"event": "test"}
    broadcast(payload)

    mock_loop.call_soon_threadsafe.assert_called_once()

    # Obtém e fecha a corrotina para evitar o RuntimeWarning de unawaited coroutine
    coro = mock_loop.call_soon_threadsafe.call_args[0][1]
    coro.close()