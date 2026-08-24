"""
離線資料前處理主流程：
    抓取刑法全文 -> 智慧分塊 -> BGE-M3 向量化 -> 寫入 PostgreSQL(pgvector)

使用方式：
    # 正式使用：從法務部官方資料來源抓取最新全文
    python -m ingest.ingest_pipeline

    # 開發/離線測試：使用內建的少量示範條文，不需要網路
    python -m ingest.ingest_pipeline --use-sample

    # 強制重新下載官方資料（略過本機快取）
    python -m ingest.ingest_pipeline --force-download
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from db.db_utils import get_connection, run_schema  # noqa: E402
from ingest.chunker import LawChunk, chunk_law  # noqa: E402
from ingest.embedder import embed_texts  # noqa: E402
from ingest.fetch_law import (  # noqa: E402
    ParsedLaw,
    fetch_and_parse,
    load_parsed_law,
)

SAMPLE_LAW_JSON = Path(__file__).resolve().parent.parent / "data" / "sample_criminal_code.json"
SCHEMA_SQL = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def _load_sample_law() -> ParsedLaw:
    print(f"[ingest_pipeline] 使用離線示範資料：{SAMPLE_LAW_JSON}（非最新版本，僅供測試）")
    return load_parsed_law(SAMPLE_LAW_JSON)


def _guess_article_no_sort(article_no: str) -> int | None:
    """從「第 361-2 條」這種格式取出排序用的主條號數字（361）。

    次條號（-2 的部分）不編進排序值，只用主條號排序；同一主條號底下的
    -1、-2……仍然會排在一起，只是彼此之間的先後順序不保證，這對於「依條號
    瀏覽/排序」的使用情境已經足夠，不需要做到小數點級的精確排序。
    """
    match = re.search(r"第\s*([0-9]+)", article_no or "")
    return int(match.group(1)) if match else None


def upsert_chunks(chunks: list[LawChunk], batch_size: int = 16) -> None:
    if not chunks:
        print("[ingest_pipeline] 沒有任何 chunk 需要寫入，略過。")
        return

    # 這次傳進來的 chunks 視為「這部法規（law_pcode）的完整最新狀態」。
    delete_sql = "DELETE FROM law_chunks WHERE law_pcode = %(law_pcode)s;"
    upsert_sql = """
        INSERT INTO law_chunks (
            law_name, law_pcode, law_modified_date, source_url,
            article_no, article_no_sort, chapter_path, content, chunk_index, chunk_count, embedding
        ) VALUES (
            %(law_name)s, %(law_pcode)s, %(law_modified_date)s, %(source_url)s,
            %(article_no)s, %(article_no_sort)s, %(chapter_path)s, %(content)s, %(chunk_index)s, %(chunk_count)s, %(embedding)s
        )
        ON CONFLICT (law_pcode, article_no, chunk_index)
        DO UPDATE SET
            content = EXCLUDED.content,
            article_no_sort = EXCLUDED.article_no_sort,
            chapter_path = EXCLUDED.chapter_path,
            law_modified_date = EXCLUDED.law_modified_date,
            source_url = EXCLUDED.source_url,
            chunk_count = EXCLUDED.chunk_count,
            embedding = EXCLUDED.embedding,
            updated_at = now();
    """

    law_pcode = chunks[0].law_pcode

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 【重要】單純用 UPSERT 只會覆蓋 chunk_index 對得上的舊資料，不會刪除多出來的。
            # 若條文修法後變短、chunk 數變少（例如某條從切 3 塊縮成 1 塊），多出來的
            # chunk_index=1、2 會變成永久殘留的孤兒資料（實測驗證過），之後檢索有機率
            # 撈到過期內容。因此改成「整部法規全刪再重建」：
            #   1) 先刪掉這個 law_pcode 底下所有舊資料
            #   2) 再把這次解析出來的全部 chunk 重新寫入
            # 兩步驟包在同一個交易裡（要到最後才 commit），任一步失敗都會整個 rollback，
            # 不會留下「刪了舊的但新的還沒寫完」的半殘狀態。
            cur.execute(delete_sql, {"law_pcode": law_pcode})
            deleted_count = cur.rowcount

            for i in tqdm(range(0, len(chunks), batch_size), desc="寫入資料庫"):
                batch = chunks[i : i + batch_size]
                vectors = embed_texts([c.content for c in batch])
                for chunk_obj, vector in zip(batch, vectors):
                    cur.execute(
                        upsert_sql,
                        {
                            "law_name": chunk_obj.law_name,
                            "law_pcode": chunk_obj.law_pcode,
                            "law_modified_date": chunk_obj.law_modified_date,
                            "source_url": chunk_obj.source_url,
                            "article_no": chunk_obj.article_no,
                            "article_no_sort": _guess_article_no_sort(chunk_obj.article_no),
                            "chapter_path": chunk_obj.chapter_path,
                            "content": chunk_obj.content,
                            "chunk_index": chunk_obj.chunk_index,
                            "chunk_count": chunk_obj.chunk_count,
                            "embedding": vector,
                        },
                    )
            conn.commit()

    print(f"[ingest_pipeline] 已清除該法規舊資料 {deleted_count} 筆，寫入新資料 {len(chunks)} 筆")


def run(use_sample: bool = False, force_download: bool = False, apply_schema: bool = True) -> None:
    if apply_schema:
        run_schema(SCHEMA_SQL)

    if use_sample:
        law = _load_sample_law()
    else:
        law = fetch_and_parse(force_download=force_download)

    print(f"[ingest_pipeline] 法規：{law.law_name}（{law.law_pcode}），共 {len(law.articles)} 條")
    chunks = chunk_law(law)
    print(f"[ingest_pipeline] 分塊完成，共 {len(chunks)} 個 chunk，開始向量化並寫入資料庫...")
    upsert_chunks(chunks)
    print("[ingest_pipeline] 完成！")
    print(f"[ingest_pipeline] 資料來源：{law.source_url}")
    print(f"[ingest_pipeline] 法規版本（最新異動日期）：{law.law_modified_date or '未提供'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="刑法查詢助手 - 資料前處理主流程")
    parser.add_argument("--use-sample", action="store_true", help="使用內建示範條文，不需網路，僅供開發測試")
    parser.add_argument("--force-download", action="store_true", help="忽略快取，重新下載官方法規資料檔")
    parser.add_argument("--skip-schema", action="store_true", help="略過建立/更新資料庫 schema 的步驟")
    args = parser.parse_args()

    run(
        use_sample=args.use_sample,
        force_download=args.force_download,
        apply_schema=not args.skip_schema,
    )
