import torch

# ================= CÂMERA =================
RTSP_URL = 0  # 0 = webcam local | "rtsp://..." para câmera IP


# ================= MODELO =================
MODEL_R3D = "melhor_modelo_tcc_final.pth"

# Quantidade de frames utilizados pelo R3D-18
BUFFER_SIZE = 16

# Threshold utilizado para considerar uma inferência
# como evidência de violência.
#
# 0.35 = 35%
# Usado neste momento para teste de sensibilidade.
VIOLENCE_THRESHOLD = 0.35

# Quantidade mínima de pessoas para ativar
# a análise de violência.
MIN_PESSOAS_VIOLENCIA = 2


# ================= EVENTOS =================
EVENTS_DIR = "eventos"

# Tempo mínimo entre dois eventos registrados
SAVE_COOLDOWN_SECONDS = 10


# ================= WEBSOCKET =================
WS_HOST = "0.0.0.0"
WS_PORT = 8000


# ================= BANCO DE DADOS =================
DATABASE_PATH = "vigia.db"


# ================= HARDWARE =================
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)