import pytest
from app.database.event_repository import (
    insert_evento,
    get_all_eventos,
    get_evento_by_id,
    get_eventos_paginados,
)


def test_insert_evento_sucesso(mocker):
    # Setup dos Mocks
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    mock_cursor.lastrowid = 1
    mock_conn.cursor.return_value = mock_cursor
    
    mocker.patch("app.database.event_repository.get_connection", return_value=mock_conn)

    # Execução
    event_id = insert_evento(
        timestamp=1710000000.0,
        datetime_str="2026-09-15 16:00:00",
        confianca=0.9543,
        pessoas=2,
        imagem="base64_img_data"
    )

    # Asserções
    assert event_id == 1
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()
    
    # Verifica se arredondou a confiança (0.9543 * 100 -> 95.43)
    mock_cursor.execute.assert_called_once()
    args, kwargs = mock_cursor.execute.call_args
    assert args[1] == (1710000000.0, "2026-09-15 16:00:00", 95.43, 2, "base64_img_data")


def test_get_all_eventos(mocker):
    # Setup dos Mocks
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    
    # Simula linhas do SQLite convertíveis em dict
    fake_rows = [
        {"id": 1, "confianca": 95.0, "timestamp": 1710000000.0},
        {"id": 2, "confianca": 88.5, "timestamp": 1710000010.0}
    ]
    mock_cursor.fetchall.return_value = fake_rows
    mock_conn.cursor.return_value = mock_cursor

    mocker.patch("app.database.event_repository.get_connection", return_value=mock_conn)

    # Execução
    resultado = get_all_eventos()

    # Asserções
    assert len(resultado) == 2
    assert resultado[0]["id"] == 1
    mock_conn.close.assert_called_once()


def test_get_evento_by_id_encontrado(mocker):
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    fake_row = {"id": 10, "confianca": 92.0, "pessoas": 1}
    mock_cursor.fetchone.return_value = fake_row
    mock_conn.cursor.return_value = mock_cursor

    mocker.patch("app.database.event_repository.get_connection", return_value=mock_conn)

    resultado = get_evento_by_id(10)

    assert resultado == fake_row
    mock_conn.close.assert_called_once()


def test_get_evento_by_id_nao_encontrado(mocker):
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn.cursor.return_value = mock_cursor

    mocker.patch("app.database.event_repository.get_connection", return_value=mock_conn)

    resultado = get_evento_by_id(999)

    assert resultado is None
    mock_conn.close.assert_called_once()


def test_get_eventos_paginados(mocker):
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    
    fake_rows = [{"id": 1}, {"id": 2}]
    fake_count = {"total": 15}
    
    mock_cursor.fetchall.return_value = fake_rows
    mock_cursor.fetchone.return_value = fake_count
    mock_conn.cursor.return_value = mock_cursor

    mocker.patch("app.database.event_repository.get_connection", return_value=mock_conn)

    # Execução para a página 2 com 5 itens por página (offset deve ser 5)
    resultado = get_eventos_paginados(page=2, per_page=5)

    # Asserções
    assert resultado["page"] == 2
    assert resultado["per_page"] == 5
    assert resultado["total"] == 15
    assert len(resultado["eventos"]) == 2
    mock_conn.close.assert_called_once()