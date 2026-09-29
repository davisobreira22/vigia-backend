from app.database.connection import get_connection


def insert_evento(timestamp: float, datetime_str: str, confianca: float, pessoas: int, imagem: str):
    """Insere um evento de violência detectado no banco de dados."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO eventos (timestamp, datetime, confianca, pessoas, imagem)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, datetime_str, round(confianca * 100, 2), pessoas, imagem))

    conn.commit()
    event_id = cursor.lastrowid
    conn.close()

    print(f"💾 Evento #{event_id} salvo no banco. Confiança: {round(confianca * 100, 2)}%")
    return event_id


def get_all_eventos():
    """Retorna todos os eventos ordenados do mais recente para o mais antigo."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM eventos ORDER BY timestamp DESC")
    rows = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return rows


def get_evento_by_id(event_id: int):
    """Retorna um evento específico pelo ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM eventos WHERE id = ?", (event_id,))
    row = cursor.fetchone()

    conn.close()
    return dict(row) if row else None


def get_eventos_paginados(page: int = 1, per_page: int = 20):
    """Retorna eventos paginados."""
    offset = (page - 1) * per_page
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM eventos ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (per_page, offset)
    )
    rows = [dict(row) for row in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) as total FROM eventos")
    total = cursor.fetchone()["total"]

    conn.close()
    return {"eventos": rows, "total": total, "page": page, "per_page": per_page}
