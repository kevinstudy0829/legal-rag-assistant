"""
對外的主要入口函式：answer_question(user_query) -> AnswerResult

流程：
    1. 將使用者問題向量化
    2. 檢索（向量 or 混合檢索，依 settings.ENABLE_HYBRID_SEARCH）取回候選條文
    3. （選用）用交叉編碼器重排序，取最相關的 TOP_K_FINAL 筆
    4. 組 prompt，呼叫 Gemini 產生回答
    5. 回傳答案 + 引用來源 + 免責聲明（已內含於 answer 文字中）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from ingest.embedder import embed_query  # noqa: E402
from query.generator import AnswerResult, generate_answer  # noqa: E402
from query.retriever import RetrievedChunk, retrieve  # noqa: E402


def retrieve_top_chunks(user_query: str) -> list[RetrievedChunk]:
    """檢索階段：向量/混合檢索 → （選用）reranker 重排序，取最相關的 TOP_K_FINAL 筆。

    拆成獨立函式是為了給 API 的 streaming 端點用：這段是 CPU 密集、無法真正
    串流的部分，呼叫端可以把它丟到背景執行緒跑、期間自己送心跳；等這段做完、
    確定要送去給 Gemini 的候選條文後，才進入可以真正逐字串流的生成階段。
    """
    if not user_query or not user_query.strip():
        raise ValueError("使用者問題不可為空")

    query_embedding = embed_query(user_query)
    candidates = retrieve(user_query, query_embedding, top_k=settings.TOP_K_CANDIDATES)

    if settings.ENABLE_RERANK and candidates:
        from query.reranker import rerank

        return rerank(user_query, candidates, top_k=settings.TOP_K_FINAL)
    return candidates[: settings.TOP_K_FINAL]


def answer_question(user_query: str) -> AnswerResult:
    top_chunks = retrieve_top_chunks(user_query)
    return generate_answer(user_query, top_chunks)


if __name__ == "__main__":
    demo_query = "我朋友欠我錢不還，我趁他不注意把他放在桌上的手機拿走抵債，這樣算什麼罪？"
    result = answer_question(demo_query)
    print("=" * 70)
    print(f"問題：{demo_query}")
    print("=" * 70)
    print(result.answer)
    print("\n[參考來源]")
    for src in result.sources:
        print(f"- {src.law_name} {src.article_no}（分數: {src.score:.4f}）")
