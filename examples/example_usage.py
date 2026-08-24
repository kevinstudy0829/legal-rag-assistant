"""
使用範例：展示兩種呼叫方式
    1. 直接呼叫 Python 函式（適合整合進其他 Python 專案）
    2. 呼叫 FastAPI 端點（適合前端、其他語言的服務呼叫）

執行前請確認：
    1. 已完成 db/schema.sql 建表、ingest/ingest_pipeline.py 匯入資料
    2. .env 已設定 DATABASE_URL 與 GEMINI_API_KEY
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

EXAMPLE_QUESTIONS = [
    "我在網路上看到有人賣假的演唱會門票，付款後才發現對方根本沒有票，直接把我封鎖，這是什麼罪？",
    "鄰居半夜一直放很吵的音樂，我很生氣就跑去他家門口大聲辱罵他三字經，這樣我會有法律責任嗎？",
    "同事離職前把公司客戶名單複製一份帶走，交給新公司使用，這樣算什麼罪？",
]


def example_direct_function_call() -> None:
    from query.qa_engine import answer_question

    print("=" * 70)
    print("範例一：直接呼叫 Python 函式")
    print("=" * 70)

    for q in EXAMPLE_QUESTIONS[:1]:
        result = answer_question(q)
        print(f"\n問題：{q}\n")
        print(result.answer)
        print("\n[參考來源]")
        for src in result.sources:
            print(f"  - {src.law_name} {src.article_no}")


def example_http_api_call() -> None:
    import requests

    print("\n" + "=" * 70)
    print("範例二：呼叫 FastAPI 端點（需先執行 `uvicorn api:app --reload`）")
    print("=" * 70)

    url = "http://localhost:8000/ask"
    for q in EXAMPLE_QUESTIONS[1:2]:
        try:
            resp = requests.post(url, json={"question": q}, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"\n[略過] 無法連線到 {url}，請先啟動 API 服務：uvicorn api:app --reload")
            return
        data = resp.json()
        print(f"\n問題：{q}\n")
        print(data["answer"])
        print("\n[參考來源]")
        for src in data["sources"]:
            print(f"  - {src['law_name']} {src['article_no']}")


if __name__ == "__main__":
    example_direct_function_call()
    example_http_api_call()
