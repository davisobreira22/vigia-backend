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
    BUFFER_SIZE,
    VIOLENCE_THRESHOLD,
    MIN_PESSOAS_VIOLENCIA,
    DEVICE
)

from app.engine import ViolenceDetector

from app.services.event_service import save_event
from app.database.event_repository import get_all_eventos


# ==============================================================
# CONFIGURAÇÃO TEMPORAL
# ==============================================================

# O treinamento usou 16 frames distribuídos pelo vídeo.
# Na câmera vamos guardar mais frames e retirar 16 uniformemente.

CAMERA_BUFFER_SIZE = 32

# Janela para evitar disparos isolados
DECISION_WINDOW = 5

# Quantas inferências positivas dentro da janela
# são necessárias para confirmar violência.
MIN_VIOLENCE_HITS = 2

# Tempo mínimo entre eventos
EVENT_COOLDOWN = 10


def start_inference(
    frame_queue: Queue,
    broadcast_fn,
    model_path: str
):

    # ==========================================================
    # R3D-18
    # ==========================================================

    print("🧠 Carregando R3D-18...")

    violence_model = ViolenceDetector(
        model_path,
        device=DEVICE
    )

    print("👁️ Carregando YOLO...")

    yolo_model = YOLO(
        "yolov8n.pt"
    )

    print("✅ YOLO carregado.")

    # ==========================================================
    # BUFFERS
    # ==========================================================

    frame_buffer = deque(
        maxlen=CAMERA_BUFFER_SIZE
    )

    violence_scores = deque(
        maxlen=DECISION_WINDOW
    )

    # ==========================================================
    # CONTROLE DE EVENTOS
    # ==========================================================

    last_event_time = 0

    violencia_confirmada = False

    # ==========================================================
    # LOG INICIAL
    # ==========================================================

    print()
    print("==============================================")
    print("🚀 VIGIA — PIPELINE DE INFERÊNCIA")
    print("==============================================")
    print(f"🧠 Dispositivo: {DEVICE}")
    print(f"🎥 Buffer temporal: {CAMERA_BUFFER_SIZE} frames")
    print("🎯 Frames enviados ao R3D: 16")
    print(f"🎯 Threshold violência: {VIOLENCE_THRESHOLD:.2f}")
    print(f"👥 Mínimo de pessoas: {MIN_PESSOAS_VIOLENCIA}")
    print(f"🪟 Janela temporal: {DECISION_WINDOW}")
    print(f"🚨 Acionamento: {MIN_VIOLENCE_HITS}/{DECISION_WINDOW}")
    print("👁️ Confiança YOLO: 0.25")
    print("==============================================")
    print()

    # ==========================================================
    # LOOP PRINCIPAL
    # ==========================================================

    while True:

        try:

            frame = frame_queue.get(
                timeout=0.05
            )

        except Exception:
            continue

        if frame is None:
            continue

        # ======================================================
        # CÓPIAS
        # ======================================================

        original_frame = frame.copy()

        display_frame = frame.copy()

        # ======================================================
        # ADICIONA FRAME AO BUFFER
        # ======================================================

        frame_buffer.append(
            original_frame
        )

        timestamp = time.time()

        # ======================================================
        # VARIÁVEIS
        # ======================================================

        pessoas = 0

        conf = 0.0

        estado = "SAFE"

        violencia = False

        box_color = (
            0,
            255,
            0
        )

        yolo_boxes = []

        # ======================================================
        # YOLO — DETECÇÃO DE PESSOAS
        # ======================================================

        try:

            results = yolo_model(
                original_frame,
                classes=[0],
                conf=0.25,
                iou=0.45,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:

                for box in result.boxes:

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    yolo_conf = float(
                        box.conf[0]
                    )

                    pessoas += 1

                    yolo_boxes.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": yolo_conf
                    })

        except Exception as e:

            print(
                f"⚠️ Erro no YOLO: {e}"
            )

            pessoas = 0
            yolo_boxes = []

        # ======================================================
        # R3D-18
        # ======================================================

        if (
            pessoas >= MIN_PESSOAS_VIOLENCIA
            and len(frame_buffer) >= CAMERA_BUFFER_SIZE
        ):

            try:

                # ==================================================
                # SELECIONA 16 FRAMES UNIFORMEMENTE
                #
                # Isso aproxima o comportamento do treinamento:
                #
                # np.linspace(..., 16)
                # ==================================================

                buffer_frames = list(
                    frame_buffer
                )

                indices = np.linspace(
                    0,
                    len(buffer_frames) - 1,
                    16,
                    dtype=int
                )

                selected_frames = [
                    buffer_frames[i]
                    for i in indices
                ]

                # ==================================================
                # PRÉ-PROCESSAMENTO
                # ==================================================

                tensor = (
                    violence_model
                    .process_sequence(
                        selected_frames
                    )
                )

                if tensor is not None:

                    # ==================================================
                    # PREDIÇÃO
                    # ==================================================

                    _, conf = (
                        violence_model.predict(
                            tensor,
                            threshold=VIOLENCE_THRESHOLD
                        )
                    )

                    conf = float(conf)

                    # ==================================================
                    # GUARDA SCORE
                    # ==================================================

                    violence_scores.append(
                        conf
                    )

                    # ==================================================
                    # CALCULA QUANTAS INFERÊNCIAS
                    # PASSARAM DO THRESHOLD
                    # ==================================================

                    violence_hits = sum(
                        score >= VIOLENCE_THRESHOLD
                        for score in violence_scores
                    )

                    # ==================================================
                    # CONFIRMA VIOLÊNCIA
                    # ==================================================

                    if (
                        len(violence_scores)
                        >= DECISION_WINDOW
                        and
                        violence_hits
                        >= MIN_VIOLENCE_HITS
                    ):

                        violencia_confirmada = True

                    else:

                        violencia_confirmada = False

                    # ==================================================
                    # ESTADO
                    # ==================================================

                    if violencia_confirmada:

                        estado = "VIOLÊNCIA"

                        violencia = True

                        box_color = (
                            0,
                            0,
                            255
                        )

                    elif conf >= 0.25:

                        estado = "PERIGO"

                        violencia = False

                        box_color = (
                            0,
                            255,
                            255
                        )

                    else:

                        estado = "SAFE"

                        violencia = False

                        box_color = (
                            0,
                            255,
                            0
                        )

                    # ==================================================
                    # LOG
                    # ==================================================

                    print(
                        f"🧠 R3D-18 | "
                        f"Pessoas: {pessoas} | "
                        f"Score violência: "
                        f"{conf * 100:.2f}% | "
                        f"Janela: "
                        f"{violence_hits}/"
                        f"{len(violence_scores)} | "
                        f"Estado: {estado}"
                    )

                    # ==================================================
                    # REGISTRO DO EVENTO
                    # ==================================================

                    current_time = time.time()

                    if (
                        violencia_confirmada
                        and
                        (
                            current_time
                            - last_event_time
                            >= EVENT_COOLDOWN
                        )
                    ):

                        last_event_time = (
                            current_time
                        )

                        threading.Thread(
                            target=_safe_save_event,
                            args=(
                                original_frame.copy(),
                                conf,
                                pessoas,
                                timestamp
                            ),
                            daemon=True
                        ).start()

                        print(
                            "🚨 EVENTO DE "
                            "VIOLÊNCIA REGISTRADO"
                        )

                        # Limpa janela depois do evento
                        violence_scores.clear()

            except Exception as e:

                print(
                    f"❌ Erro na inferência "
                    f"R3D-18: {e}"
                )

                estado = "SAFE"

                violencia = False

                violencia_confirmada = False

                conf = 0.0

                box_color = (
                    0,
                    255,
                    0
                )

        else:

            # ==================================================
            # MENOS DE 2 PESSOAS
            # ==================================================

            estado = "SAFE"

            violencia = False

            violencia_confirmada = False

            conf = 0.0

            box_color = (
                0,
                255,
                0
            )

            violence_scores.clear()

        # ==========================================================
        # DESENHA PESSOAS
        # ==========================================================

        for person in yolo_boxes:

            x1 = person["x1"]
            y1 = person["y1"]
            x2 = person["x2"]
            y2 = person["y2"]

            yolo_conf = person["conf"]

            # ------------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )

            # ------------------------------------------------------
            # LABEL
            # ------------------------------------------------------

            label = (
                f"Pessoa "
                f"{yolo_conf * 100:.0f}% "
                f"| {estado}"
            )

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            label_y = max(
                text_height + baseline + 5,
                y1
            )

            cv2.rectangle(
                display_frame,
                (
                    x1,
                    label_y
                    - text_height
                    - baseline
                    - 5
                ),
                (
                    x1
                    + text_width
                    + 8,
                    label_y
                ),
                box_color,
                -1
            )

            cv2.putText(
                display_frame,
                label,
                (
                    x1 + 4,
                    label_y - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        # ==========================================================
        # INFORMAÇÕES NA IMAGEM
        # ==========================================================

        cv2.putText(
            display_frame,
            f"Pessoas: {pessoas}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display_frame,
            f"IA: {estado}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display_frame,
            f"Score: {conf * 100:.1f}%",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
            cv2.LINE_AA
        )

        # ==========================================================
        # CONVERTE FRAME PARA BASE64
        # ==========================================================

        try:

            success, buffer_img = cv2.imencode(
                ".jpg",
                display_frame
            )

            if success:

                frame_b64 = base64.b64encode(
                    buffer_img
                ).decode("utf-8")

            else:

                frame_b64 = ""

        except Exception as e:

            print(
                f"⚠️ Erro ao codificar "
                f"frame: {e}"
            )

            frame_b64 = ""

        # ==========================================================
        # LATÊNCIA
        # ==========================================================

        latencia_ms = int(
            (
                time.time()
                - timestamp
            ) * 1000
        )

        # ==========================================================
        # TOTAL DE EVENTOS
        # ==========================================================

        try:

            total_eventos = len(
                get_all_eventos()
            )

        except Exception:

            total_eventos = 0

        # ==========================================================
        # ENVIA PARA FRONTEND
        # ==========================================================

        try:

            broadcast_fn({

                "timestamp": timestamp,

                "pessoasCount": pessoas,

                "violencia": violencia,

                "iaScore": conf,

                "latenciaMs": latencia_ms,

                "totalEventos": total_eventos,

                "estado": estado,

                "frame": frame_b64

            })

        except Exception as e:

            print(
                f"⚠️ Erro ao enviar "
                f"frame para frontend: {e}"
            )


# ==============================================================
# SALVAR EVENTO SEM TRAVAR INFERÊNCIA
# ==============================================================

def _safe_save_event(
    frame,
    conf,
    pessoas,
    timestamp
):

    try:

        save_event(
            frame,
            conf,
            pessoas,
            timestamp
        )

    except Exception as e:

        print(
            f"⚠️ Erro ao salvar "
            f"evento de violência: {e}"
        )