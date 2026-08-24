"""
取得《中華民國刑法》全文。

【重要】為何不直接爬取 law.moj.gov.tw/LawClass/LawAll.aspx？
------------------------------------------------------------
1. 該頁面的 robots.txt 明確禁止自動化存取。
2. 法務部已透過「政府資料開放平臺」提供官方、結構化、允許重製利用的
   法規資料檔（政府資料開放授權條款–第1版，免費），不需要爬蟲：
      - 資料集頁面: https://data.gov.tw/dataset/18289 （中文法規_法律資料檔下載）
      - 檔案下載（RAW XML）:
        https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=RAW_XML&AuData=CF
   這個檔案包含「法律」層級的全部法規（含刑法），每月更新一次。

本模組流程：
    1. 下載上述官方 XML 壓縮/原始檔（若本機已有暫存檔且未過期則直接複用）。
    2. 在 XML 中找出 LawName 等於「中華民國刑法」的那個 <Law> 節點。
    3. 將章節標題（ArticleType = "C"）與條文（ArticleType = "A"）整理成
       一份乾淨的中繼資料 + 條文清單，供後續 chunker.py 使用。

【提醒】上面提到「禁止自動化存取」指的是 law.moj.gov.tw 的網頁版查詢介面
（LawAll.aspx），其 robots.txt 是 `Disallow: /`。sendlaw.moj.gov.tw 這個
開放資料下載端點是完全不同的網域，且 data.gov.tw 資料集頁面明確標示為
「政府資料開放授權條款–第1版」、免費、供程式化下載重製利用，因此
`download_bulk_xml()` 用 `requests.get()` 直接下載並無疑慮。

僅存的不確定性是：XML 標籤名稱（法規名稱 / 編章節 / 條號 / 條文內容 等）
未來若因官方調整資料集格式而改變，程式可能需要跟著調整候選標籤清單
（`_LAW_NODE_TAGS` 等）。程式已盡量寫得寬容，但仍建議每次正式匯入前先跑
一次 `python -m ingest.fetch_law --inspect`，人工核對抓到的條文數量與
內容是否正確、是否為最新版本。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_XML_CACHE = DATA_DIR / "law_bulk_raw.xml"
PARSED_LAW_JSON = DATA_DIR / "criminal_code.json"

# 候選標籤名稱：官方 XML 欄位命名歷來大致穩定，但仍以「多個候選、取第一個命中」
# 的寫法增加韌性，避免未來欄位命名微調就整支程式失效。
_LAW_NODE_TAGS = ["Law", "LawData", "法規"]
_LAW_NAME_TAGS = ["LawName", "法規名稱"]
_LAW_URL_TAGS = ["LawURL", "法規網址"]
_LAW_MODIFIED_TAGS = ["LawModifiedDate", "最新異動日期"]
_ARTICLES_CONTAINER_TAGS = ["LawArticles", "法規內容"]
# 章節標題節點：文字直接放在標籤內（如 <編章節>第 一 編 總則</編章節>），
# 與條文節點是同層的兄弟節點，靠「標籤名稱本身」區分，而不是共用同一標籤 + 內部欄位判斷。
_CHAPTER_HEADING_TAGS = ["編章節"]
_ARTICLE_NODE_TAGS = ["LawArticle", "條文"]
# 相容舊格式用：某些資料集是「同一種節點 + ArticleType 欄位」區分章節(C)/條文(A)
_ARTICLE_TYPE_TAGS = ["ArticleType"]
_ARTICLE_NO_TAGS = ["ArticleNo", "條號"]
_ARTICLE_CONTENT_TAGS = ["ArticleContent", "條文內容"]


@dataclass
class LawArticle:
    article_no: str          # 例如「第 271 條」
    content: str              # 條文內容（不含章節標題）
    chapter_path: str = ""    # 例如「第二編 分則 / 第二十二章 殺人罪」


@dataclass
class ParsedLaw:
    law_name: str
    law_pcode: str
    law_modified_date: str
    source_url: str
    articles: list[LawArticle] = field(default_factory=list)


def _first_text(node: ET.Element, candidate_tags: list[str]) -> str:
    for tag in candidate_tags:
        found = node.find(tag)
        if found is not None and found.text:
            return found.text.strip()
    return ""


def _extract_pcode_from_url(url: str) -> str:
    match = re.search(r"[Pp][Cc]ode=([A-Za-z0-9]+)", url or "")
    return match.group(1) if match else ""


def download_bulk_xml(force: bool = False) -> Path:
    """下載法務部官方「中文法規_法律資料檔」原始 XML，並快取到 data/ 目錄。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XML_CACHE.exists() and not force:
        print(f"[fetch_law] 使用本機快取：{RAW_XML_CACHE}")
        return RAW_XML_CACHE

    print(f"[fetch_law] 下載官方法規資料檔：{settings.LAW_BULK_XML_URL}")
    resp = requests.get(settings.LAW_BULK_XML_URL, timeout=120)
    resp.raise_for_status()
    RAW_XML_CACHE.write_bytes(resp.content)
    print(f"[fetch_law] 已下載並快取至：{RAW_XML_CACHE}（{len(resp.content) / 1_048_576:.1f} MB）")
    return RAW_XML_CACHE


def _iter_law_nodes(root: ET.Element):
    for tag in _LAW_NODE_TAGS:
        nodes = root.findall(f".//{tag}")
        if nodes:
            yield from nodes
            return


def _iter_content_nodes(law_node: ET.Element):
    """依 XML 原始順序，依序 yield 章節標題節點與條文節點。

    這兩種節點在官方 XML 中是同一層的兄弟節點（都直接在 <法規內容> 底下），
    順序本身就代表「這一條屬於哪一章」。因此不能像舊版那樣只挑出條文節點，
    否則章節標題節點會被整個濾掉，之後每一條的 chapter_path 永遠是空字串。
    """
    container = None
    for container_tag in _ARTICLES_CONTAINER_TAGS:
        container = law_node.find(container_tag)
        if container is not None:
            break
    # 找不到獨立容器就退回直接在 law_node 底下找（相容沒有外層容器的資料格式）
    parent = container if container is not None else law_node
    yield from parent


def parse_law_from_bulk_xml(xml_path: Path, target_law_name: str) -> ParsedLaw:
    """從整包法律資料 XML 中，取出指定法規名稱（如「中華民國刑法」）的條文。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    matched_law_node = None
    for law_node in _iter_law_nodes(root):
        name = _first_text(law_node, _LAW_NAME_TAGS)
        if name == target_law_name:
            matched_law_node = law_node
            break

    if matched_law_node is None:
        raise ValueError(
            f"在官方法規資料檔中找不到法規名稱完全等於「{target_law_name}」的資料。\n"
            "可能原因：1) XML 標籤名稱與程式假設不同 2) 法規名稱有變動。\n"
            "請執行 `python -m ingest.fetch_law --inspect` 檢查實際的 XML 結構，"
            "並視情況調整 ingest/fetch_law.py 中的候選標籤清單。"
        )

    law_url = _first_text(matched_law_node, _LAW_URL_TAGS) or settings.LAW_SOURCE_PAGE_URL
    parsed = ParsedLaw(
        law_name=target_law_name,
        law_pcode=_extract_pcode_from_url(law_url) or "C0000001",
        law_modified_date=_first_text(matched_law_node, _LAW_MODIFIED_TAGS),
        source_url=law_url,
    )

    current_chapter_stack: list[str] = []  # 例如 ["第二編 分則", "第二十二章 殺人罪"]

    for node in _iter_content_nodes(matched_law_node):
        tag = node.tag

        if tag in _CHAPTER_HEADING_TAGS:
            # 章節標題節點：文字直接在節點內，如 <編章節>第 一 編 總則</編章節>。
            # 更新目前的章節路徑（同層或更高層新標題會截斷舊路徑，見 _update_chapter_stack）。
            heading = (node.text or "").strip()
            if heading:
                current_chapter_stack = _update_chapter_stack(current_chapter_stack, heading)
            continue

        if tag in _ARTICLE_NODE_TAGS:
            content = _first_text(node, _ARTICLE_CONTENT_TAGS)
            if not content:
                continue
            article_no = _first_text(node, _ARTICLE_NO_TAGS) or _guess_article_no(content)
            parsed.articles.append(
                LawArticle(
                    article_no=article_no,
                    content=content,
                    chapter_path=" / ".join(current_chapter_stack),
                )
            )
            continue

        # 相容舊格式：同一種節點 + ArticleType 欄位（C=章節 / 其他=條文）
        article_type = _first_text(node, _ARTICLE_TYPE_TAGS)
        content = _first_text(node, _ARTICLE_CONTENT_TAGS)
        if not content:
            continue
        if article_type == "C":
            current_chapter_stack = _update_chapter_stack(current_chapter_stack, content)
            continue
        article_no = _first_text(node, _ARTICLE_NO_TAGS) or _guess_article_no(content)
        parsed.articles.append(
            LawArticle(
                article_no=article_no,
                content=content,
                chapter_path=" / ".join(current_chapter_stack),
            )
        )

    if not parsed.articles:
        raise ValueError(
            f"找到了「{target_law_name}」節點，但未解析出任何條文。"
            "請用 --inspect 模式檢查 ArticleType/ArticleContent 標籤是否正確。"
        )

    return parsed


def _update_chapter_stack(stack: list[str], heading: str) -> list[str]:
    """粗略地依「編/章/節/款/目」關鍵字判斷層級，同層或更高層新標題會截斷舊路徑。"""
    level_keywords = ["編", "章", "節", "款", "目"]
    level = next((i for i, kw in enumerate(level_keywords) if kw in heading), len(level_keywords))
    new_stack = stack[:level]
    new_stack.append(heading)
    return new_stack


def _guess_article_no(content: str) -> str:
    match = re.match(r"^(第\s*[0-9〇一二三四五六七八九十百千]+(?:之[0-9一二三四五六七八九十]+)?\s*條)", content)
    return match.group(1) if match else "未知條號"


def save_parsed_law(parsed: ParsedLaw, out_path: Path = PARSED_LAW_JSON) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "law_name": parsed.law_name,
        "law_pcode": parsed.law_pcode,
        "law_modified_date": parsed.law_modified_date,
        "source_url": parsed.source_url,
        "article_count": len(parsed.articles),
        "articles": [
            {"article_no": a.article_no, "content": a.content, "chapter_path": a.chapter_path}
            for a in parsed.articles
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fetch_law] 已解析 {len(parsed.articles)} 條，寫入 {out_path}")
    return out_path


def load_parsed_law(path: Path = PARSED_LAW_JSON) -> ParsedLaw:
    data = json.loads(path.read_text(encoding="utf-8"))
    articles = [LawArticle(**a) for a in data["articles"]]
    return ParsedLaw(
        law_name=data["law_name"],
        law_pcode=data["law_pcode"],
        law_modified_date=data["law_modified_date"],
        source_url=data["source_url"],
        articles=articles,
    )


def fetch_and_parse(force_download: bool = False) -> ParsedLaw:
    xml_path = download_bulk_xml(force=force_download)
    parsed = parse_law_from_bulk_xml(xml_path, settings.LAW_NAME)
    save_parsed_law(parsed)
    return parsed


def _inspect(xml_path: Path) -> None:
    """人工核對用：印出第一個匹配到的 <Law> 節點的原始 XML 片段。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"[inspect] 根節點標籤：<{root.tag}>，共有 {len(list(root))} 個子節點")
    sample = next(iter(_iter_law_nodes(root)), None)
    if sample is None:
        print("[inspect] 找不到任何符合候選標籤的 <Law> 節點，請人工開啟 XML 檔案確認結構。")
        return
    print(f"[inspect] 範例法規名稱：{_first_text(sample, _LAW_NAME_TAGS)}")
    print("[inspect] 前 2000 字元的原始 XML：")
    print(ET.tostring(sample, encoding="unicode")[:2000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下載並解析《中華民國刑法》全文")
    parser.add_argument("--force-download", action="store_true", help="忽略本機快取，重新下載")
    parser.add_argument("--inspect", action="store_true", help="僅印出 XML 結構供人工核對，不寫入 JSON")
    args = parser.parse_args()

    xml_file = download_bulk_xml(force=args.force_download)
    if args.inspect:
        _inspect(xml_file)
    else:
        result = parse_law_from_bulk_xml(xml_file, settings.LAW_NAME)
        save_parsed_law(result)
        print(f"[fetch_law] 完成。法規版本（最新異動日期）：{result.law_modified_date or '未提供'}")
        print(f"[fetch_law] 官方來源：{result.source_url}")
