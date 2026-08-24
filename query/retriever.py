"""
檢索模組：
    - vector_search(): 純向量（cosine 距離）檢索
    - keyword_search(): 以 pg_trgm 做關鍵字（字元 n-gram）檢索，對中文不需額外斷詞套件
    - hybrid_search(): 用 Reciprocal Rank Fusion (RRF) 融合以上兩種排名結果

為什麼中文關鍵字檢索用 pg_trgm 而不是 PostgreSQL 內建全文檢索(tsvector)？
    PostgreSQL 內建的全文檢索斷詞器預設不支援中文分詞（會需要額外安裝 zhparser
    等擴充套件）。pg_trgm 在字元層級做 3-gram 相似度比對，雖然不是「語意」比對，
    但對中文關鍵字（例如使用者輸入「偷東西」「詐騙」等字詞）有不錯的召回效果，
    且不需要額外系統相依套件，部署更簡單。若你的環境已安裝 zhparser，也可以把
    keyword_search 替換成 tsvector 版本以取得更精準的結果。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from db.db_utils import get_connection  # noqa: E402


@dataclass
class RetrievedChunk:
    id: int
    law_name: str
    law_pcode: str
    article_no: str
    chapter_path: str
    content: str
    source_url: str
    law_modified_date: str
    score: float  # 依檢索方法不同，可能是 cosine 相似度、trigram 相似度，或 RRF 融合分數


_SELECT_COLUMNS = """
    id, law_name, law_pcode, article_no, chapter_path, content, source_url, law_modified_date
"""


def vector_search(query_embedding: np.ndarray, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.TOP_K_CANDIDATES
    sql = f"""
        SELECT {_SELECT_COLUMNS}, 1 - (embedding <=> %(qvec)s) AS score
        FROM law_chunks
        ORDER BY embedding <=> %(qvec)s
        LIMIT %(top_k)s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"qvec": query_embedding, "top_k": top_k})
            rows = cur.fetchall()
    return [_row_to_chunk(r) for r in rows]


def keyword_search(query_text: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.TOP_K_CANDIDATES
    sql = f"""
        SELECT {_SELECT_COLUMNS}, similarity(content, %(qtext)s) AS score
        FROM law_chunks
        WHERE content %% %(qtext)s
        ORDER BY score DESC
        LIMIT %(top_k)s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"qtext": query_text, "top_k": top_k})
            rows = cur.fetchall()
    return [_row_to_chunk(r) for r in rows]


def hybrid_search(
    query_text: str,
    query_embedding: np.ndarray,
    top_k: int | None = None,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """用 Reciprocal Rank Fusion 融合向量檢索與關鍵字檢索的排名。

    RRF 分數公式： score(d) = sum( 1 / (rrf_k + rank_i(d)) )，rank 由 1 起算。
    這是資訊檢索中常見、不需要調權重、對兩種分數尺度不同也很穩健的融合方法。
    """
    top_k = top_k or settings.TOP_K_CANDIDATES
    vector_results = vector_search(query_embedding, top_k=settings.TOP_K_CANDIDATES)
    keyword_results = keyword_search(query_text, top_k=settings.TOP_K_CANDIDATES)

    rrf_scores: dict[int, float] = {}
    chunk_by_id: dict[int, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        chunk_by_id[chunk.id] = chunk

    for rank, chunk in enumerate(keyword_results, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        chunk_by_id.setdefault(chunk.id, chunk)

    ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    results = []
    for cid in ranked_ids:
        chunk = chunk_by_id[cid]
        chunk.score = rrf_scores[cid]
        results.append(chunk)
    return results


def retrieve(query_text: str, query_embedding: np.ndarray, top_k: int | None = None) -> list[RetrievedChunk]:
    """統一入口：依設定決定要用純向量檢索還是混合檢索。"""
    if settings.ENABLE_HYBRID_SEARCH:
        return hybrid_search(query_text, query_embedding, top_k=top_k)
    return vector_search(query_embedding, top_k=top_k)


def _row_to_chunk(row: tuple) -> RetrievedChunk:
    (id_, law_name, law_pcode, article_no, chapter_path, content, source_url, law_modified_date, score) = row
    return RetrievedChunk(
        id=id_,
        law_name=law_name,
        law_pcode=law_pcode,
        article_no=article_no,
        chapter_path=chapter_path or "",
        content=content,
        source_url=source_url or "",
        law_modified_date=law_modified_date or "",
        score=float(score),
    )
