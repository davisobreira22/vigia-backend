import os
import json
import pytest
from app.services import event_service
from app.services.event_service import save_event, _update_index_json


@pytest.fixture(autouse=True)
def reset_cooldown():
    """Garante que a variável de cooldown seja resetada antes de cada teste."""
    event_service._last_save_time = 0


def test_save_event_sucesso(mocker, tmp_path):
    mock_cv2 = mocker.patch("app.services.event_service.cv2.imwrite")
    mock_insert = mocker.patch("app.services.event_service.insert_evento")
    mocker.patch("app.services.event_service.EVENTS_DIR", str(tmp_path))

    ts = 1710000000.0  # Exemplo de timestamp
    save_event(frame="fake_frame", conf=0.95, pessoas=2, timestamp=ts)

    mock_cv2.assert_called_once()
    mock_insert.assert_called_once()

    # Verifica se criou o index.json corretamente
    index_file = tmp_path / "index.json"
    assert index_file.exists()

    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["confianca"] == 95.0
        assert data[0]["pessoas"] == 2


def test_save_event_respeita_cooldown(mocker, tmp_path):
    mocker.patch("app.services.event_service.cv2.imwrite")
    mock_insert = mocker.patch("app.services.event_service.insert_evento")
    mocker.patch("app.services.event_service.EVENTS_DIR", str(tmp_path))
    mocker.patch("app.services.event_service.SAVE_COOLDOWN_SECONDS", 5)

    ts = 1710000000.0
    
    # Primeira chamada salva
    save_event("frame1", 0.9, 1, ts)
    assert mock_insert.call_count == 1

    # Segunda chamada imediata é bloqueada pelo cooldown
    save_event("frame2", 0.9, 1, ts)
    assert mock_insert.call_count == 1


def test_update_index_json_com_arquivo_corrompido(mocker, tmp_path):
    mocker.patch("app.services.event_service.EVENTS_DIR", str(tmp_path))
    
    # Cria arquivo index.json com JSON inválido para forçar o try/except
    index_file = tmp_path / "index.json"
    index_file.write_text("invalid json content")

    _update_index_json(
        timestamp=1710000000.0,
        datetime_str="15/09/2026 16:00:00",
        conf=0.88,
        pessoas=1,
        base_name="evento_teste"
    )

    # Verifica se o arquivo foi sobrescrito recuperando da falha
    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["confianca"] == 88.0