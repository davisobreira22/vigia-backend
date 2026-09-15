import cv2
import os
import json
import time
from datetime import datetime

from config import EVENTS_DIR, SAVE_COOLDOWN_SECONDS
from app.database.event_repository import insert_evento

os.makedirs(EVENTS_DIR, exist_ok=True)

_last_save_time = 0


def save_event(frame, conf: float, pessoas: int, timestamp: float):
    """
    Salva o frame do evento em disco (JPG) e persiste os metadados
    no banco de dados SQLite. Aplica cooldown para evitar duplicatas.
    """
    global _last_save_time

    now = time.time()
    if now - _last_save_time < SAVE_COOLDOWN_SECONDS:
        return

    _last_save_time = now

    # Salva imagem em disco
    ts_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    base_name = f"evento_{ts_str}"
    img_path = os.path.join(EVENTS_DIR, f"{base_name}.jpg")
    cv2.imwrite(img_path, frame)

    datetime_str = datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S")

    # Persiste no banco de dados
    insert_evento(
        timestamp=timestamp,
        datetime_str=datetime_str,
        confianca=conf,
        pessoas=pessoas,
        imagem=f"{base_name}.jpg"
    )

    # Mantém compatibilidade: atualiza index.json local também
    _update_index_json(timestamp, datetime_str, conf, pessoas, base_name)


def _update_index_json(timestamp, datetime_str, conf, pessoas, base_name):
    """Mantém o index.json local para compatibilidade com o frontend atual."""
    index_path = os.path.join(EVENTS_DIR, "index.json")
    existing = []

    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append({
        "timestamp": timestamp,
        "datetime": datetime_str,
        "confianca": round(conf * 100, 2),
        "pessoas": pessoas,
        "imagem": f"{base_name}.jpg"
    })

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
