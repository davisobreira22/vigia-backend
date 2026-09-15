import cv2
from queue import Queue
from config import RTSP_URL


def start_capture(frame_queue: Queue):
    """
    Captura frames da câmera e coloca na fila.
    Roda em thread dedicada.
    """
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"❌ Não foi possível abrir a câmera: {RTSP_URL}")

    print(f"📷 Câmera iniciada: {RTSP_URL}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)
