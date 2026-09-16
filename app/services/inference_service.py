import cv2
import torch
import time
import base64
import threading
import numpy as np
from queue import Queue
from collections import deque
from ultralytics import YOLO

from config import (
    BUFFER_SIZE, VIOLENCE_THRESHOLD,
    MIN_PESSOAS_VIOLENCIA, DEVICE
)
from app.engine import ViolenceDetector
from app.services.event_service import save_event


def start_inference(frame_queue: Queue, broadcast_fn, model_path: str):
    """
    Pipeline principal de inferência.
    1. Lê frames da fila
    2. Roda YOLO para contar pessoas
    3. Se >= MIN_PESSOAS_VIOLENCIA, roda R3D-18
    4. Faz broadcast do frame + metadados padronizados para o WebSocket
    Roda em thread dedicada.
    """
    violence_model = ViolenceDetector(model_path, device=DEVICE)
    yolo_model = YOLO("yolov8n.pt")
    buffer = deque(maxlen=BUFFER_SIZE)

    print(f"🧠 Inference pipeline iniciado em: {DEVICE}")

    while True:
        try:
            frame = frame_queue.get(timeout=0.05)
        except Exception:
            continue

        buffer.append(frame)
        timestamp = time.time()

        # --- Etapa 1: YOLO (Detecção de Pessoas) ---
        results = yolo_model(frame, classes=[0], verbose=False)
        pessoas = int(len(results[0].boxes)) if results[0].boxes is not None else 0

        violencia = False
        conf = 0.0

        # --- Etapa 2: R3D-18 (Inferência de Violência) ---
        if pessoas >= MIN_PESSOAS_VIOLENCIA and len(buffer) == BUFFER_SIZE:
            tensor = _build_tensor(buffer)
            
            # Desativa o cálculo de gradientes para economizar GPU/RAM
            with torch.no_grad():
                pred, conf = violence_model.predict(tensor, threshold=VIOLENCE_THRESHOLD)
            
            violencia = bool(pred == 1)

            if violencia:
                # Dispara gravação do evento sem bloquear a inferência
                threading.Thread(
                    target=_safe_save_event,
                    args=(frame.copy(), float(conf), pessoas, timestamp),
                    daemon=True
                ).start()

        # --- Etapa 3: Codificação do Frame ---
        _, buffer_img = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer_img).decode('utf-8')

        # Calculo de latência em milissegundos
        latencia_ms = int((time.time() - timestamp) * 1000)

        # --- Etapa 4: Broadcast padronizado ---
        broadcast_fn({
            "timestamp": timestamp,
            "pessoasCount": pessoas,
            "violencia": violencia,
            "iaScore": float(conf),
            "latenciaMs": latencia_ms,
            "frame": frame_b64
        })


def _safe_save_event(frame, conf, pessoas, timestamp):
    """Encapsula a gravação do evento para capturar exceções sem quebrar a thread."""
    try:
        save_event(frame, conf, pessoas, timestamp)
    except Exception as e:
        print(f"⚠️ Erro ao salvar evento de violência: {e}")


def _build_tensor(buffer: deque) -> torch.Tensor:
    """Converte o buffer de frames em tensor para o R3D-18."""
    frames_resized = [cv2.resize(f, (112, 112)) for f in buffer]
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_resized]
    frames_np = np.array(frames_rgb)

    tensor = torch.tensor(frames_np).float() / 255.0
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)

    if DEVICE == "cuda":
        tensor = tensor.half()

    return tensor