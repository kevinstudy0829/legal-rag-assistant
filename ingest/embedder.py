"""
使用 BGE-M3（BAAI/bge-m3）將文字轉為 1024 維向量。

BGE-M3 同時支援 dense / sparse / multi-vector 三種表示法，本專案只使用其中
最通用的 dense 向量（1024 維、cosine 相似度），已足以應付條文檢索的需求；
如果之後想做更講究的混合檢索，也可以擴充成同時儲存 sparse 向量。

注意：BGE-M3 不像部分 BGE 英文版模型需要在 query 前加上特殊指令前綴
（例如 "Represent this sentence for searching relevant passages:"），
query 與 document 用同一種方式編碼即可。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] 載入向量化模型：{settings.EMBEDDING_MODEL_NAME}（device={settings.EMBEDDING_DEVICE}）")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=settings.EMBEDDING_DEVICE)
    return _model


def embed_texts(texts: Iterable[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
    """將一批文字轉成 (n, 1024) 的 numpy 陣列，已做 L2 normalize，方便直接用 cosine 距離比對。"""
    model = get_embedding_model()
    texts = list(texts)
    if not texts:
        return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    if embeddings.shape[1] != settings.EMBEDDING_DIM:
        raise ValueError(
            f"向量維度不符：模型輸出 {embeddings.shape[1]} 維，"
            f"但設定檔 EMBEDDING_DIM={settings.EMBEDDING_DIM}。"
            "請確認 EMBEDDING_MODEL_NAME 是否仍為 BGE-M3，或同步更新 db/schema.sql 中的 vector(1024)。"
        )
    return embeddings.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """方便查詢端使用的單筆版本，回傳 shape=(1024,) 的向量。"""
    return embed_texts([text])[0]


if __name__ == "__main__":
    sample_vecs = embed_texts(["殺人者，處死刑、無期徒刑或十年以上有期徒刑。", "竊取他人財物"], show_progress=True)
    print(f"[embedder] 測試向量 shape: {sample_vecs.shape}")
