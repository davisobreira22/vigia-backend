
import asyncio
import base64
import json
import os

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import WS_HOST, WS_PORT


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="VIGIA API")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRETÓRIO DE EVENTOS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

EVENTOS_DIR = os.path.join(
    BASE_DIR,
    "eventos"
)

os.makedirs(
    EVENTOS_DIR,
    exist_ok=True
)


# Permite acessar as imagens:
#
# http://localhost:8000/eventos/nome_da_imagem.jpg
#
app.mount(
    "/eventos",
    StaticFiles(directory=EVENTOS_DIR),
    name="eventos"
)


# ============================================================
# CLIENTES WEBSOCKET
# ============================================================

_clients = set()

_loop = None

_frame_queue = None


# ============================================================
# FILA DE FRAMES
# ============================================================

def set_frame_queue(frame_queue):
    """
    Registra a fila que recebe os frames enviados
    pelo navegador.
    """

    global _frame_queue

    _frame_queue = frame_queue


# ============================================================
# API - EVENTOS
# ============================================================

@app.get("/api/events")
def get_events():
    """
    Retorna os eventos registrados no index.json.

    O sistema VIGIA salva os eventos em:

        eventos/index.json

    e as imagens correspondentes em:

        eventos/evento_XXXXXX.jpg
    """

    index_path = os.path.join(
        EVENTOS_DIR,
        "index.json"
    )

    try:

        # ----------------------------------------------------
        # VERIFICA SE O INDEX EXISTE
        # ----------------------------------------------------

        if not os.path.exists(index_path):

            print(
                f"⚠️ Arquivo de eventos não encontrado: "
                f"{index_path}"
            )

            return []


        # ----------------------------------------------------
        # LÊ O INDEX.JSON
        # ----------------------------------------------------

        with open(
            index_path,
            "r",
            encoding="utf-8"
        ) as file:

            events = json.load(file)


        # ----------------------------------------------------
        # GARANTE QUE É UMA LISTA
        # ----------------------------------------------------

        if not isinstance(events, list):

            print(
                "⚠️ index.json não contém uma lista de eventos."
            )

            return []


        # ----------------------------------------------------
        # REMOVE EVENTOS INVÁLIDOS
        # ----------------------------------------------------

        valid_events = []

        for event in events:

            if not isinstance(event, dict):

                continue

            imagem = event.get("imagem")

            if not imagem:

                continue

            valid_events.append(event)


        # ----------------------------------------------------
        # MAIS RECENTES PRIMEIRO
        # ----------------------------------------------------

        valid_events.sort(
            key=lambda event: float(
                event.get(
                    "timestamp",
                    0
                )
            ),
            reverse=True
        )


        print(
            f"📋 API /api/events: "
            f"{len(valid_events)} evento(s) encontrado(s)."
        )


        return valid_events


    except json.JSONDecodeError as exc:

        print(
            f"❌ Erro ao ler index.json: {exc}"
        )

        return []


    except Exception as exc:

        print(
            f"❌ Erro ao carregar eventos: {exc}"
        )

        return []


# ============================================================
# API - HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "VIGIA",
        "websocket": True
    }


# ============================================================
# WEBSOCKET
# ============================================================

async def _handler(websocket: WebSocket):
    """
    Recebe frames enviados pelo frontend.
    """

    await websocket.accept()

    _clients.add(websocket)

    print(
        "🔌 Cliente WebSocket conectado."
    )

    try:

        while True:

            msg = await websocket.receive_text()

            # ------------------------------------------------
            # PING
            # ------------------------------------------------

            if msg == "ping":

                await websocket.send_text(
                    "pong"
                )

                continue


            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            try:

                data = json.loads(msg)

            except (
                TypeError,
                json.JSONDecodeError
            ):

                continue


            # ------------------------------------------------
            # FRAME
            # ------------------------------------------------

            if data.get("type") != "frame":

                continue


            raw = data.get("frame")

            if not raw:

                continue


            if _frame_queue is None:

                print(
                    "⚠️ Frame recebido, "
                    "mas a fila ainda não foi configurada."
                )

                continue


            # ------------------------------------------------
            # DECODIFICAÇÃO
            # ------------------------------------------------

            try:

                image_bytes = base64.b64decode(
                    raw,
                    validate=True
                )

                array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    array,
                    cv2.IMREAD_COLOR
                )

                if frame is None:

                    continue


                # ------------------------------------------------
                # MANTÉM SOMENTE OS FRAMES MAIS RECENTES
                # ------------------------------------------------

                while _frame_queue.full():

                    try:

                        _frame_queue.get_nowait()

                    except Exception:

                        break


                try:

                    _frame_queue.put_nowait(
                        frame
                    )

                except Exception:

                    pass


            except Exception as exc:

                print(
                    f"⚠️ Frame inválido recebido: {exc}"
                )


    except WebSocketDisconnect:

        pass


    except Exception as exc:

        print(
            f"⚠️ Erro no WebSocket: {exc}"
        )


    finally:

        _clients.discard(
            websocket
        )

        print(
            "🔌 Cliente WebSocket desconectado."
        )


# ============================================================
# ROTAS WEBSOCKET
# ============================================================

@app.websocket("/")
async def websocket_root(
    websocket: WebSocket
):

    await _handler(
        websocket
    )


@app.websocket("/ws")
async def websocket_ws(
    websocket: WebSocket
):

    await _handler(
        websocket
    )


# ============================================================
# BROADCAST
# ============================================================

async def _send_to_all(
    msg: str
):

    if not _clients:

        return


    disconnected = set()


    for ws in list(_clients):

        try:

            await ws.send_text(
                msg
            )

        except Exception:

            disconnected.add(
                ws
            )


    _clients.difference_update(
        disconnected
    )


def broadcast(data: dict):
    """
    Envia os resultados da IA para todos
    os clientes WebSocket conectados.
    """

    if not _clients:

        return


    if _loop is None:

        return


    msg = json.dumps(
        data,
        ensure_ascii=False
    )


    try:

        asyncio.run_coroutine_threadsafe(
            _send_to_all(msg),
            _loop
        )


    except Exception as exc:

        print(
            f"⚠️ Erro ao transmitir resultado: {exc}"
        )


# ============================================================
# SERVIDOR HTTP + WEBSOCKET
# ============================================================

def run_ws_server(frame_queue):
    """
    Inicia o servidor FastAPI/Uvicorn.

    A mesma porta 8000 atende:

    HTTP:
        /api/events
        /api/health
        /eventos

    WebSocket:
        /
        /ws
    """

    global _loop
    global _frame_queue


    # --------------------------------------------------------
    # REGISTRA A FILA RECEBIDA DO MAIN.PY
    # --------------------------------------------------------

    _frame_queue = frame_queue


    # --------------------------------------------------------
    # CONFIGURA UVICORN
    # --------------------------------------------------------

    config = uvicorn.Config(
        app,
        host=WS_HOST,
        port=WS_PORT,
        log_level="warning",
    )


    server = uvicorn.Server(
        config
    )


    # --------------------------------------------------------
    # EXECUÇÃO
    # --------------------------------------------------------

    async def run():

        global _loop

        _loop = asyncio.get_running_loop()


        print(
            f"🌐 HTTP + WebSocket rodando em "
            f"http://{WS_HOST}:{WS_PORT}"
        )

        print(
            f"📡 WebSocket: "
            f"ws://{WS_HOST}:{WS_PORT}"
        )

        print(
            f"📊 API eventos: "
            f"http://{WS_HOST}:{WS_PORT}/api/events"
        )

        print(
            f"🖼️ Imagens: "
            f"http://{WS_HOST}:{WS_PORT}/eventos"
        )


        await server.serve()


    asyncio.run(
        run()
    )
