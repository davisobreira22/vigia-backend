import cv2
import time
from queue import Queue
from config import RTSP_URL


def start_capture(frame_queue: Queue):
    """
    Captura frames da câmera e coloca na fila.
    Trata auto-reconexão e detecta webcam USB vs Stream RTSP.
    Roda em thread dedicada.
    """
    # Se RTSP_URL for numérico (ex: "0" em string), converte para int (Webcam USB)
    source = int(RTSP_URL) if isinstance(RTSP_URL, str) and RTSP_URL.isdigit() else RTSP_URL

    while True:
        # Usa CAP_DSHOW apenas se for webcam local (int) no Windows
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"❌ Não foi possível abrir a fonte de vídeo: {RTSP_URL}. Tentando em 3s...")
            time.sleep(3)
            continue

        print(f"📷 Câmera conectada e ativa: {RTSP_URL}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Falha ao ler frame da câmera. Tentando reconectar...")
                break

            # Mantém apenas os frames mais recentes limpos na fila
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass

            frame_queue.put(frame)

        cap.release()
        time.sleep(1)