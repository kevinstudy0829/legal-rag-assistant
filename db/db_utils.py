"""
PostgreSQL 連線輔助工具。統一在這裡註冊 pgvector 型別，
其餘模組一律透過 get_connection() 取得連線。
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402


@contextlib.contextmanager
def get_connection(register_vector_type: bool = True):
    """回傳一個連線（context manager，會自動關閉）。

    預設會呼叫 register_vector() 註冊 pgvector 型別，但這要求資料庫裡已經
    執行過 `CREATE EXTENSION vector`。run_schema() 本身就是負責建立這個
    擴充套件的地方，在全新資料庫上執行時擴充套件還不存在，此時若仍呼叫
    register_vector() 會直接丟出 `vector type not found in the database`，
    形成雞生蛋問題，因此 run_schema() 會傳入 register_vector_type=False。
    """
    conn = psycopg2.connect(settings.DATABASE_URL)
    try:
        if register_vector_type:
            register_vector(conn)
        yield conn
    finally:
        conn.close()


def run_schema(schema_path: str | Path) -> None:
    """執行 schema.sql，建立資料表與索引（可重複執行，具備 IF NOT EXISTS 保護）。"""
    sql = Path(schema_path).read_text(encoding="utf-8")
    with get_connection(register_vector_type=False) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print(f"[db_utils] Schema 已套用：{schema_path}")


if __name__ == "__main__":
    # 方便直接執行： python -m db.db_utils
    schema_file = Path(__file__).resolve().parent / "schema.sql"
    run_schema(schema_file)
