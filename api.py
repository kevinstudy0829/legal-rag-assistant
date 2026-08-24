"""
選用的 HTTP API 介面（FastAPI），同時 serve 靜態網頁前端（static/index.html）。

啟動方式：
    uvicorn api:app --reload --port 8000
    然後在瀏覽器開啟 http://localhost:8000/ 即可使用網頁介面

呼叫範例（純 API）：
    curl -X POST http://localhost:8000/ask \\
         -H "Content-Type: application/json" \\
         -d '{"question": "我拿刀威脅便利商店店員交出收銀機裡的錢，這樣算什麼罪？"}'
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import AsyncIterator, Callable, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from query.generator import DISCLAIMER, build_prompt, stream_gemini
from query.qa_engine import retrieve_top_chunks

# 查詢一次通常要 20~90 秒（本地跑 embedding/reranker + 呼叫 Gemini），這段期間
# 如果完全沒有資料在連線上傳輸，很容易被行動網路的電信商 NAT、或手機上的
# 本地 VPN App（例如 AdGuard 的廣告過濾功能）誤判成閒置連線而提早砍斷
#（實測案例：手機用 WiFi + AdGuard，每次都準時在 30 秒被斷線；電腦或關掉
# AdGuard 後就正常）。/ask 用 streaming response 因應：
#   1. 檢索 + reranker 這段是 CPU 密集、無法真正逐字輸出的，用心跳撐著連線；
#   2. 進入 Gemini 生成階段後改成「真的逐字串流」，本身就會持續有資料流動，
#      不需要額外心跳，順便讓使用者看到像聊天機器人一樣逐字跑出來的效果。
_HEARTBEAT_INTERVAL_SECONDS = 8

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="刑法查詢助手 API",
    description="基於 RAG 架構、根據《中華民國刑法》條文回答法律問題的查詢助手。本服務僅供輔助參考，不構成正式法律意見。",
    version="1.0.0",
)

# 開發用途放寬 CORS：網頁前端與 API 目前預設同源（都是這個 FastAPI app），
# 但保留寬鬆設定方便你之後把前端另外部署（例如純靜態網站託管）時仍可呼叫這支 API。
# 正式環境建議把 allow_origins 改成明確列出你的前端網域，而不是用 "*"。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="使用者描述的行為或想詢問的問題")


class SourceItem(BaseModel):
    law_name: str
    article_no: str
    chapter_path: str
    source_url: str
    score: float


# 註：/ask 改成 streaming response 之後，FastAPI 沒辦法再用 response_model 自動
# 驗證/產生 OpenAPI schema（StreamingResponse 的內容型別是純文字），下面這個
# AskResponse 純粹保留下來當作「最終 JSON payload 長什麼樣子」的文件用途。


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    """網頁前端首頁：static/index.html。"""
    return FileResponse(STATIC_DIR / "index.html")


async def _iterate_blocking_generator(sync_gen_factory: Callable[[], Iterator[str]]) -> AsyncIterator[str]:
    """把一個「同步、會阻塞」的產生器橋接成 async generator。

    google-genai 的 streaming API（stream_gemini）底層是同步、阻塞式的 HTTP
    streaming，直接在事件迴圈裡疊代會卡住整個 uvicorn worker。這裡另外開一條
    背景執行緒真正去跑那個同步產生器，每 yield 一個項目就丟進 queue，
    async 這邊再從 queue 裡撈出來繼續 yield，藉此不擋住事件迴圈。
    """
    q: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def worker() -> None:
        try:
            for item in sync_gen_factory():
                q.put(("item", item))
        except Exception as exc:  # noqa: BLE001
            q.put(("error", exc))
        finally:
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    loop = asyncio.get_running_loop()
    while True:
        kind, payload = await loop.run_in_executor(None, q.get)
        if kind == "item":
            yield payload  # type: ignore[misc]
        elif kind == "error":
            raise payload  # type: ignore[misc]
        else:
            break


def _sources_payload(chunks) -> list[dict]:
    return [
        {
            "law_name": c.law_name,
            "article_no": c.article_no,
            "chapter_path": c.chapter_path,
            "source_url": c.source_url,
            "score": c.score,
        }
        for c in chunks
    ]


async def _ask_stream(question: str) -> AsyncIterator[str]:
    """NDJSON streaming：一行一個 JSON 事件，前端逐行解析。

    事件型別（"type" 欄位）：
        sources    —— 檢索完成，附上這次引用的參考條文清單（只送一次）
        delta      —— Gemini 生成的文字片段（可能送很多次，前端累加顯示）
        disclaimer —— 強制附加的免責聲明（不是 Gemini 生成的，最後單獨送一次）
        error      —— 發生錯誤，附上錯誤說明

    檢索 + reranker 階段是 CPU 密集、無法真正逐字輸出，先用心跳（空行）撐著
    連線；進入 Gemini 生成階段後，文字片段本身就會持續流動，不需要心跳。
    """
    loop = asyncio.get_running_loop()
    retrieve_task = loop.run_in_executor(None, retrieve_top_chunks, question)

    while not retrieve_task.done():
        yield "\n"  # 心跳：純空行，前端解析時會直接忽略
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)

    try:
        chunks = retrieve_task.result()
    except ValueError as exc:
        yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
        return
    except Exception as exc:  # noqa: BLE001
        yield json.dumps({"type": "error", "detail": f"查詢過程發生錯誤：{exc}"}) + "\n"
        return

    yield json.dumps({"type": "sources", "sources": _sources_payload(chunks)}) + "\n"

    if not chunks:
        fallback = (
            "很抱歉，目前的條文資料庫中找不到與您描述之行為明顯相關的刑法條文，"
            "無法據以分析可能觸犯的罪名。建議您換個方式描述行為細節，或直接洽詢律師。"
        )
        yield json.dumps({"type": "delta", "text": fallback}) + "\n"
        yield json.dumps({"type": "disclaimer", "text": DISCLAIMER}) + "\n"
        return

    prompt = build_prompt(question, chunks)

    try:
        async for delta in _iterate_blocking_generator(lambda: stream_gemini(prompt)):
            yield json.dumps({"type": "delta", "text": delta}) + "\n"
    except Exception as exc:  # noqa: BLE001
        yield json.dumps({"type": "error", "detail": f"生成回答時發生錯誤：{exc}"}) + "\n"
        return

    yield json.dumps({"type": "disclaimer", "text": DISCLAIMER}) + "\n"


@app.post("/ask")
async def ask(payload: AskRequest) -> StreamingResponse:
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="使用者問題不可為空")

    return StreamingResponse(
        _ask_stream(payload.question),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},  # 提示 nginx 這條路由不要緩衝，立即把每個 chunk 轉發出去
    )
