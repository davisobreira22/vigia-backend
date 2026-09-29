import pytest
import numpy as np
import torch
from queue import Queue
from app.services.inference_service import start_inference, _build_tensor


class StopLoop(BaseException):
    """Exceção customizada derivada de BaseException para escapar do 'except Exception:'."""
    pass


def test_start_inference_timeout_e_loop(mocker):
    mocker.patch("app.services.inference_service.ViolenceDetector")
    mocker.patch("app.services.inference_service.YOLO")
    
    mock_queue = mocker.MagicMock()
    # 1ª chamada simula timeout da fila; 2ª interrompe o loop
    mock_queue.get.side_effect = [Exception("Timeout"), StopLoop()]

    mock_broadcast = mocker.MagicMock()

    with pytest.raises(StopLoop):
        start_inference(mock_queue, mock_broadcast, "fake_model.pth")


def test_start_inference_sem_pessoas_suficientes(mocker):
    mocker.patch("app.services.inference_service.ViolenceDetector")
    
    mock_yolo_instance = mocker.MagicMock()
    mock_result = mocker.MagicMock()
    mock_result.boxes = []
    mock_yolo_instance.return_value = [mock_result]
    mocker.patch("app.services.inference_service.YOLO", return_value=mock_yolo_instance)

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    mock_queue = mocker.MagicMock()
    mock_queue.get.side_effect = [fake_frame, StopLoop()]

    mock_broadcast = mocker.MagicMock()

    with pytest.raises(StopLoop):
        start_inference(mock_queue, mock_broadcast, "fake_model.pth")

    mock_broadcast.assert_called_once()
    payload = mock_broadcast.call_args[0][0]
    assert payload["pessoas"] == 0
    assert payload["violencia"] is False


def test_start_inference_detecta_violencia(mocker):
    mock_detector_instance = mocker.MagicMock()
    mock_detector_instance.predict.return_value = (1, 0.95)
    mocker.patch("app.services.inference_service.ViolenceDetector", return_value=mock_detector_instance)

    mock_yolo_instance = mocker.MagicMock()
    mock_result = mocker.MagicMock()
    mock_result.boxes = [mocker.MagicMock(), mocker.MagicMock()]
    mock_yolo_instance.return_value = [mock_result]
    mocker.patch("app.services.inference_service.YOLO", return_value=mock_yolo_instance)

    mocker.patch("app.services.inference_service.BUFFER_SIZE", 1)
    mocker.patch("app.services.inference_service.MIN_PESSOAS_VIOLENCIA", 2)
    mocker.patch("app.services.inference_service.save_event")
    
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_queue = mocker.MagicMock()
    mock_queue.get.side_effect = [fake_frame, StopLoop()]

    mock_broadcast = mocker.MagicMock()

    with pytest.raises(StopLoop):
        start_inference(mock_queue, mock_broadcast, "fake_model.pth")

    mock_broadcast.assert_called_once()
    payload = mock_broadcast.call_args[0][0]
    assert payload["pessoas"] == 2
    assert payload["violencia"] is True
    assert payload["confianca"] == 0.95


def test_build_tensor(mocker):
    mocker.patch("app.services.inference_service.DEVICE", "cpu")
    
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    buffer = [fake_frame, fake_frame]

    tensor = _build_tensor(buffer)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 2, 112, 112)