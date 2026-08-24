"""
互動式命令列介面（CLI）。

使用方式：
    python app.py

輸入行為描述後按 Enter，助手會回覆可能觸犯的法條分析；輸入 exit / quit 離開。
"""
from __future__ import annotations

from query.qa_engine import answer_question

BANNER = """
========================================================
  刑法查詢助手（RAG based on 中華民國刑法）
  本工具僅供輔助參考，不構成正式法律意見。
  輸入 'exit' 或 'quit' 離開。
========================================================
"""


def main() -> None:
    print(BANNER)
    while True:
        try:
            user_input = input("請描述您想詢問的行為或情境：\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("再見！")
            break
        if not user_input:
            continue

        print("\n分析中，請稍候...\n")
        try:
            result = answer_question(user_input)
        except Exception as exc:  # noqa: BLE001
            print(f"[錯誤] 查詢過程發生問題：{exc}\n")
            continue

        print(result.answer)
        if result.sources:
            print("\n[參考條文來源]")
            for src in result.sources:
                print(f"  - {src.law_name} {src.article_no}（相關度分數：{src.score:.4f}）｜{src.source_url}")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
