import torch

# ================= CÂMERA =================
RTSP_URL = 0  # 0 = webcam local | "rtsp://..." para câmera IP

# ================= MODELO =================
MODEL_R3D = "melhor_modelo_tcc_final_ofc.pth"
BUFFER_SIZE = 16
VIOLENCE_THRESHOLD = 0.85
MIN_PESSOAS_VIOLENCIA = 2

# ================= EVENTOS =================
EVENTS_DIR = "eventos"
SAVE_COOLDOWN_SECONDS = 10

# ================= WEBSOCKET =================
WS_HOST = "0.0.0.0"
WS_PORT = 8000

# ================= BANCO DE DADOS =================
DATABASE_PATH = "vigia.db"

# ================= HARDWARE =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
