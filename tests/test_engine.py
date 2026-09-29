import pytest
import torch
import numpy as np
from app.engine import ViolenceDetector


def test_init_sucesso(mocker):
    mocker.patch("torch.load", return_value={"state_dict": {}})
    mocker.patch("torch.nn.Module.load_state_dict")
    
    detector = ViolenceDetector("fake_path.pth", device="cpu")
    assert detector.model is not None


def test_init_cuda(mocker):
    mocker.patch("torch.load", return_value={"model_state_dict": {}})
    mocker.patch("torch.nn.Module.load_state_dict")
    mocker.patch("torch.nn.Module.to")
    mock_half = mocker.patch("torch.nn.Module.half")

    detector = ViolenceDetector("fake_path.pth", device="cuda")
    mock_half.assert_called_once()


def test_init_erro_carregamento(mocker):
    mocker.patch("torch.load", side_effect=Exception("Arquivo invalido"))

    with pytest.raises(Exception, match="Erro ao carregar modelo"):
        ViolenceDetector("fake_path.pth", device="cpu")


def test_process_sequence_tamanho_invalido(mocker):
    mocker.patch("torch.load", return_value={})
    mocker.patch("torch.nn.Module.load_state_dict")

    detector = ViolenceDetector("fake_path.pth", device="cpu")
    result = detector.process_sequence([np.zeros((100, 100, 3), dtype=np.uint8)] * 10)
    assert result is None


def test_process_sequence_sucesso(mocker):
    mocker.patch("torch.load", return_value={})
    mocker.patch("torch.nn.Module.load_state_dict")
    mocker.patch.object(ViolenceDetector, "__init__", return_value=None)

    detector = ViolenceDetector("fake_path.pth", device="cpu")
    detector.device = torch.device("cpu")
    detector.normalize = lambda x: x

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(16)]
    tensor = detector.process_sequence(frames)
    
    assert isinstance(tensor, torch.Tensor)


def test_process_sequence_cuda(mocker):
    mocker.patch("torch.load", return_value={})
    mocker.patch("torch.nn.Module.load_state_dict")
    mocker.patch.object(ViolenceDetector, "__init__", return_value=None)

    detector = ViolenceDetector("fake_path.pth", device="cuda")
    # Usa CPU no teste para evitar falha na transferência para GPU caso não haja suporte nativo
    detector.device = torch.device("cpu")
    detector.normalize = lambda x: x

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(16)]
    tensor = detector.process_sequence(frames)

    assert tensor is not None


def test_predict_tensor_none(mocker):
    mocker.patch("torch.load", return_value={})
    mocker.patch("torch.nn.Module.load_state_dict")

    detector = ViolenceDetector("fake_path.pth", device="cpu")
    pred, conf = detector.predict(None)
    
    assert pred == 0
    assert conf == 0.0


def test_predict_sucesso_com_violencia(mocker):
    mocker.patch("torch.load", return_value={})
    mocker.patch("torch.nn.Module.load_state_dict")

    detector = ViolenceDetector("fake_path.pth", device="cpu")
    
    mock_output = torch.tensor([[0.0, 3.0]])
    detector.model = mocker.MagicMock(return_value=mock_output)

    fake_tensor = torch.zeros((1, 3, 16, 224, 224))
    pred, conf = detector.predict(fake_tensor, threshold=0.75)

    assert pred == 1
    assert conf > 0.75