import pytest
from queue import Queue
from app.services.camera_service import start_capture


def test_start_capture_erro_abrir_camera(mocker):
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    queue = Queue(maxsize=1)

    with pytest.raises(RuntimeError) as exc_info:
        start_capture(queue)

    assert "Não foi possível abrir a câmera" in str(exc_info.value)


def test_start_capture_sucesso_e_fila_cheia(mocker):
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True

    # Simula 3 leituras: 
    # 1ª: Falha (ret=False) para testar o continue
    # 2ª: Sucesso -> insere na fila
    # 3ª: Sucesso -> lança exceção para quebrar o 'while True' e encerrar o teste
    mock_cap.read.side_effect = [
        (False, None),
        (True, "frame_1"),
        Exception("StopLoop"),
    ]

    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Cria uma fila de tamanho 1 para testar o fluxo completo
    queue = Queue(maxsize=1)

    with pytest.raises(Exception, match="StopLoop"):
        start_capture(queue)

    # Garante que o frame_1 entrou na fila
    assert not queue.empty()
    assert queue.get() == "frame_1"


def test_start_capture_ignora_frame_quando_fila_cheia(mocker):
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, "frame_2"),
        Exception("StopLoop"),
    ]

    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Preenche a fila antes de rodar
    queue = Queue(maxsize=1)
    queue.put("frame_existente")

    with pytest.raises(Exception, match="StopLoop"):
        start_capture(queue)

    # Garante que o frame_2 foi descartado porque a fila estava cheia
    assert queue.get() == "frame_existente"
    assert queue.empty()