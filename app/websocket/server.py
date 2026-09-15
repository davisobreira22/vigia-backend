import asyncio
import json
import websockets
from config import WS_HOST, WS_PORT

_clients = set()
_loop = None


async def _handler(websocket):
    """Registra cliente e mantém conexão ativa."""
    _clients.add(websocket)
    try:
        async for msg in websocket:
            if msg == "ping":
                await websocket.send("pong")
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


async def _ws_server_task():
    global _loop
    _loop = asyncio.get_running_loop()
    async with websockets.serve(_handler, WS_HOST, WS_PORT):
        print(f"🌐 WebSocket rodando em ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()


def run_ws_server():
    """Inicia o servidor WebSocket. Roda em thread dedicada."""
    asyncio.run(_ws_server_task())


def broadcast(data: dict):
    """Envia dados para todos os clientes conectados."""
    if not _clients or _loop is None:
        return

    msg = json.dumps(data)
    for ws in list(_clients):
        _loop.call_soon_threadsafe(asyncio.create_task, ws.send(msg))
