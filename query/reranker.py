"""
選用的重排序（re-ranking）模組。

第一階段的向量/混合檢索速度快，但用的是「雙塔（bi-encoder）」架構，準確度有
天花板；重排序用「交叉編碼器（cross-encoder）」對 (query, chunk) 這一對文字
一起餵進模型算相關分數，準確度通常更好，但速度較慢，所以只對第一階段選出的
少量候選（TOP_K_CANDIDATES）做重排序，取最終 TOP_K_FINAL 筆送進 LLM。

模型使用 BAAI/bge-reranker-v2-m3（與 BGE-M3 同系列，中文效果佳），透過
sentence-transformers 的 CrossEncoder 載入。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from query.retriever import RetrievedChunk  # noqa: E402

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        print(f"[reranker] 載入重排序模型：{settings.RERANKER_MODEL_NAME}（device={settings.RERANKER_DEVICE}）")
        _reranker = CrossEncoder(
            settings.RERANKER_MODEL_NAME,
            device=settings.RERANKER_DEVICE,
            trust_remote_code=True,
        )
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    if not chunks:
        return []

    top_k = top_k or settings.TOP_K_FINAL
    model = get_reranker()
    pairs = [(query, c.content) for c in chunks]
    raw_scores = model.predict(pairs)

    for chunk, score in zip(chunks, raw_scores):
        chunk.score = float(score)

    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return reranked[:top_k]
