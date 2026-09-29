import threading
import time
from queue import Queue

from config import MODEL_R3D, MIN_PESSOAS_VIOLENCIA, DEVICE
from app.database.connection import init_db
from app.websocket.server import run_ws_server, broadcast
from app.services.inference_service import start_inference


if __name__ == "__main__":
    print(f"🔥 VIGIA iniciando em: {DEVICE}")

    init_db()

    # Fila compartilhada entre WebSocket e inferência.
    frame_queue = Queue(maxsize=5)

    threading.Thread(
        target=run_ws_server,
        args=(frame_queue,),
        daemon=True,
        name="websocket-server",
    ).start()

    threading.Thread(
        target=start_inference,
        args=(frame_queue, broadcast, MODEL_R3D),
        daemon=True,
        name="inference",
    ).start()

    print("🚀 Sistema VIGIA ativo.")
    print(f"💡 IA de violência ativa com >= {MIN_PESSOAS_VIOLENCIA} pessoas detectadas.")
    print("📡 Aguardando frames do frontend em ws://localhost:8000")

    while True:
        time.sleep(1)
