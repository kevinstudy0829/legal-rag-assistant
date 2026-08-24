"""
將條文切成適合向量化的分塊（chunk）。

策略：
    1. 預設「以條為單位」— 一條就是一個 chunk，這對台灣法規最自然，因為使用者
       通常是想知道「某個行為觸犯哪一條」，條與條之間語意邊界清楚，切太細反而
       會失去上下文（例如某條的「前項」「但書」）。
    2. 若單一條文過長（例如條文中有很多款、項，超過 CHUNK_MAX_CHARS），才會
       再依字數切分，並保留 CHUNK_OVERLAP_CHARS 的重疊，避免關鍵資訊被切斷在
       兩個 chunk 的交界處而任一邊都撈不到完整語意。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from ingest.fetch_law import LawArticle, ParsedLaw  # noqa: E402


@dataclass
class LawChunk:
    law_name: str
    law_pcode: str
    law_modified_date: str
    source_url: str
    article_no: str
    chapter_path: str
    content: str
    chunk_index: int
    chunk_count: int


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    if overlap_chars >= max_chars:
        raise ValueError("CHUNK_OVERLAP_CHARS 必須小於 CHUNK_MAX_CHARS")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        # 盡量在中文句讀（。；！？）處切斷，避免切在句子中間；若附近找不到就直接硬切
        if end < text_len:
            boundary = _find_sentence_boundary(text, start, end)
            if boundary is not None:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= text_len:
            break
        start = end - overlap_chars if end - overlap_chars > start else end

    return [c for c in chunks if c]


def _find_sentence_boundary(text: str, start: int, soft_end: int) -> int | None:
    """在 [start, soft_end] 區間內，由後往前找最近的句讀符號，回傳其後一個字元的位置。"""
    punctuations = "。；;！？"
    search_window_start = max(start, soft_end - 100)  # 只在結尾附近 100 字內找，避免切得太短
    for i in range(soft_end - 1, search_window_start - 1, -1):
        if text[i] in punctuations:
            return i + 1
    return None


def chunk_article(law: ParsedLaw, article: LawArticle) -> list[LawChunk]:
    pieces = _split_with_overlap(
        article.content,
        max_chars=settings.CHUNK_MAX_CHARS,
        overlap_chars=settings.CHUNK_OVERLAP_CHARS,
    )
    return [
        LawChunk(
            law_name=law.law_name,
            law_pcode=law.law_pcode,
            law_modified_date=law.law_modified_date,
            source_url=law.source_url,
            article_no=article.article_no,
            chapter_path=article.chapter_path,
            content=piece,
            chunk_index=idx,
            chunk_count=len(pieces),
        )
        for idx, piece in enumerate(pieces)
    ]


def chunk_law(law: ParsedLaw) -> list[LawChunk]:
    if settings.CHUNK_OVERLAP_CHARS >= settings.CHUNK_MAX_CHARS:
        # 提早驗證設定：若在這裡才靠處理到某條特別長的條文才觸發 ValueError，
        # 匯入流程可能跑到一半才中斷，不容易第一時間定位是設定檔的問題。
        raise ValueError(
            "設定錯誤：CHUNK_OVERLAP_CHARS 必須小於 CHUNK_MAX_CHARS"
            f"（目前 CHUNK_OVERLAP_CHARS={settings.CHUNK_OVERLAP_CHARS}, "
            f"CHUNK_MAX_CHARS={settings.CHUNK_MAX_CHARS}），請檢查 .env 設定。"
        )
    all_chunks: list[LawChunk] = []
    for article in law.articles:
        all_chunks.extend(chunk_article(law, article))
    return all_chunks


if __name__ == "__main__":
    # 快速自我測試：用範例資料檢視分塊結果
    from ingest.fetch_law import load_parsed_law, PARSED_LAW_JSON

    if not PARSED_LAW_JSON.exists():
        print("[chunker] 尚未有解析好的法規 JSON，請先執行 `python -m ingest.fetch_law`")
    else:
        parsed_law = load_parsed_law(PARSED_LAW_JSON)
        result_chunks = chunk_law(parsed_law)
        print(f"[chunker] 共產生 {len(result_chunks)} 個 chunk（原始條文 {len(parsed_law.articles)} 條）")
        for c in result_chunks[:3]:
            print("-" * 60)
            print(f"{c.article_no}（{c.chunk_index + 1}/{c.chunk_count}）| 章節: {c.chapter_path}")
            print(c.content[:120])
