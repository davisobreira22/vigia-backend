import pytest

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Seta variáveis de ambiente simuladas."""
    monkeypatch.setenv("ENV", "testing")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

@pytest.fixture
def mock_yolo(mocker):
    """Evita carregar o arquivo yolov8n.pt do disco."""
    return mocker.patch("vigia.app.services.inference_service.YOLO")

@pytest.fixture
def mock_cv2(mocker):
    """Evita que o OpenCV tente conectar a um RTSP real."""
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, "frame_ficticio")
    return mocker.patch("cv2.VideoCapture", return_value=mock_cap)