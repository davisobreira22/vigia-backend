import pytest
from app.database.connection import get_connection, init_db


def test_get_connection_sucesso(mocker):
    # Mock do sqlite3.connect para não criar/abrir o arquivo real sqlite
    mock_sqlite_connect = mocker.patch("sqlite3.connect")
    mock_conn = mocker.MagicMock()
    mock_sqlite_connect.return_value = mock_conn

    # Executa a função
    conn = get_connection()

    # Asserções
    assert conn == mock_conn
    assert conn.row_factory is not None
    mock_sqlite_connect.assert_called_once()


def test_init_db_sucesso(mocker):
    # Setup dos Mocks de conexão e cursor
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    # Intercepta get_connection dentro do módulo connection
    mocker.patch("app.database.connection.get_connection", return_value=mock_conn)

    # Executa a inicialização do banco
    init_db()

    # Asserções
    mock_cursor.execute.assert_called_once()
    assert "CREATE TABLE IF NOT EXISTS eventos" in mock_cursor.execute.call_args[0][0]
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()