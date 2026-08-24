"""
組合 Prompt 並呼叫 Gemini，產生最終回答。

安全設計重點：
    1. System instruction 明確要求「僅根據提供的參考條文回答」、資訊不足要
       明講、禁止杜撰法條或刑度。
    2. 無論 LLM 有沒有乖乖照做，程式碼都會在回傳結果的最後「強制附加」一段
       法律免責聲明，不依賴 LLM 自律，避免因為 prompt 被使用者繞過而漏掉。
    3. 每個引用的參考條文都會附上條號與官方來源連結，方便使用者自行核對。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from query.retriever import RetrievedChunk  # noqa: E402

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **免責聲明**：以上內容由 AI 系統依據《中華民國刑法》條文自動生成，僅供一般性參考，"
    "不構成正式法律意見，亦不能取代律師之個案諮詢。具體案件的法律責任認定，需綜合案發事實、"
    "證據、行為人主觀意圖等因素由法院依法判斷。如有實際法律需求，請洽詢律師或法律扶助基金會"
    "（法律扶助專線：412-8518，手機加 02）。"
)

SYSTEM_INSTRUCTION = """你是一個專精台灣《中華民國刑法》的法律查詢助手。你會收到使用者描述的行為，
以及從刑法條文資料庫中檢索出來、最相關的幾條參考條文。請嚴格遵守以下規則：

1. 只能根據「參考條文」的內容作答，不可以使用你自己既有的法律知識去補充或杜撰參考條文中沒有的
   條號、要件或刑度。
2. 如果參考條文不足以判斷使用者描述的行為可能觸犯哪些法條，或者行為描述過於模糊，必須明確告知
   使用者「目前提供的參考條文不足以判斷」，並具體說明還缺少哪些關鍵資訊（例如行為人主觀意圖、
   是否既遂、有無阻卻違法事由等），不要硬給答案。
3. 回答時針對每一個你認為可能構成的罪名，都要清楚標明：
   - 對應的法條條號
   - 為什麼使用者描述的行為符合（或可能符合）該條的構成要件
   - 該條規定的法定刑度（依參考條文原文）
4. 若使用者描述的行為明顯不構成犯罪，或屬於阻卻違法（如正當防衛）等情形，也請根據參考條文說明。
5. 用詞盡量白話、有條理，但不要省略法律專有名詞的正確用法。
6. 絕對不要提供如何規避法律追訴、湮滅證據、或實施違法行為的具體操作建議。
7. 回答語言：繁體中文。
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[參考條文 {i}] {c.law_name} {c.article_no}"
            f"（章節：{c.chapter_path or '未標示'}；來源：{c.source_url}）\n{c.content}"
        )
    return "\n\n".join(lines)


def build_prompt(user_query: str, chunks: list[RetrievedChunk]) -> str:
    context_block = build_context_block(chunks)
    return f"""【參考條文】
{context_block}

【使用者描述的行為 / 問題】
{user_query}

請根據上述參考條文，分析使用者描述的行為可能觸犯哪些法條、構成要件是否符合，以及可能面臨的刑度。"""


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("尚未設定 GEMINI_API_KEY，請在 .env 中填入你的 Gemini API 金鑰。")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def call_gemini(prompt: str, system_instruction: str = SYSTEM_INSTRUCTION) -> str:
    client = get_client()
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,  # 法律分析場景希望輸出穩定、少發散
        ),
    )
    return response.text or ""


def stream_gemini(prompt: str, system_instruction: str = SYSTEM_INSTRUCTION):
    """逐段（streaming）呼叫 Gemini，一有新文字片段就 yield 出去。

    這是同步的產生器（底層 SDK 是同步、阻塞式的 HTTP streaming），呼叫端如果
    是在 async 環境下用，記得丟到執行緒裡跑，不要直接在事件迴圈裡疊代。
    """
    client = get_client()
    response_stream = client.models.generate_content_stream(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


@dataclass
class AnswerResult:
    answer: str
    sources: list[RetrievedChunk]
    prompt_used: str


def generate_answer(user_query: str, chunks: list[RetrievedChunk]) -> AnswerResult:
    if not chunks:
        answer = (
            "很抱歉，目前的條文資料庫中找不到與您描述之行為明顯相關的刑法條文，"
            "無法據以分析可能觸犯的罪名。建議您換個方式描述行為細節，或直接洽詢律師。"
            + DISCLAIMER
        )
        return AnswerResult(answer=answer, sources=[], prompt_used="")

    prompt = build_prompt(user_query, chunks)
    raw_answer = call_gemini(prompt)
    final_answer = raw_answer.strip() + DISCLAIMER
    return AnswerResult(answer=final_answer, sources=chunks, prompt_used=prompt)
