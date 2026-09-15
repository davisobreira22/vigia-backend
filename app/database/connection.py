import sqlite3
from config import DATABASE_PATH


def get_connection():
    """Retorna uma conexão com o banco de dados SQLite."""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Retorna linhas como dicionários
    return conn


def init_db():
    """Cria as tabelas se não existirem."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eventos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            datetime    TEXT    NOT NULL,
            confianca   REAL    NOT NULL,
            pessoas     INTEGER NOT NULL,
            imagem      TEXT    NOT NULL,
            criado_em   TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Banco de dados inicializado.")
