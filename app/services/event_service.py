import cv2
import os
import json
import time
import threading
from datetime import datetime

from config import EVENTS_DIR, SAVE_COOLDOWN_SECONDS
from app.database.event_repository import insert_evento

os.makedirs(EVENTS_DIR, exist_ok=True)

_last_save_time = 0
_file_lock = threading.Lock()


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

    # Formatação de datas
    ts_dt = datetime.fromtimestamp(timestamp)
    ts_str = ts_dt.strftime("%Y%m%d_%H%M%S")
    datetime_str = ts_dt.strftime("%d/%m/%Y %H:%M:%S")

    base_name = f"evento_{ts_str}"
    img_filename = f"{base_name}.jpg"
    img_path = os.path.join(EVENTS_DIR, img_filename)

    # 1. Salva a imagem em disco
    try:
        cv2.imwrite(img_path, frame)
    except Exception as e:
        print(f"❌ Erro ao salvar imagem no disco: {e}")
        return

    # 2. Persiste no banco de dados SQLite
    try:
        insert_evento(
            timestamp=timestamp,
            datetime_str=datetime_str,
            confianca=conf,
            pessoas=pessoas,
            imagem=img_filename
        )
    except Exception as e:
        print(f"❌ Erro ao salvar evento no SQLite: {e}")

    # 3. Mantém compatibilidade com index.json com Lock de Thread
    with _file_lock:
        _update_index_json(timestamp, datetime_str, conf, pessoas, img_filename)


def _update_index_json(timestamp, datetime_str, conf, pessoas, img_filename):
    """Atualiza o arquivo index.json local garantindo concorrência segura."""
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
        "imagem": img_filename
    })

    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Erro ao atualizar index.json: {e}")