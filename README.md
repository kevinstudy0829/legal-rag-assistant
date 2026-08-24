# 刑法查詢助手（RAG based Legal Q&A System）

以《中華民國刑法》為知識庫的自然語言查詢助手。使用者描述一段行為，系統會用
向量檢索（RAG）從刑法條文中找出最相關的參考條文，再交給 Gemini 分析可能觸犯
的法條與刑度。

> ⚠️ **本工具僅供輔助參考，不構成正式法律意見，也不能取代律師之個案諮詢。**
> 詳見〈法律免責聲明〉一節。

---

## 目錄

- [架構總覽](#架構總覽)
- [技術選型](#技術選型)
- [專案結構](#專案結構)
- [快速開始](#快速開始)
- [資料來源說明（重要）](#資料來源說明重要)
- [修法後如何更新資料](#修法後如何更新資料)
- [進階功能：混合檢索與重排序](#進階功能混合檢索與重排序)
- [使用範例](#使用範例)
- [部署建議](#部署建議)
- [法律免責聲明](#法律免責聲明)
- [已知限制與後續優化方向](#已知限制與後續優化方向)

---

## 架構總覽

```
離線（資料前處理）
  法務部官方法規開放資料（XML）
        │  ingest/fetch_law.py
        ▼
  解析出《中華民國刑法》條文
        │  ingest/chunker.py（依條分塊，過長再切分+overlap）
        ▼
  ingest/embedder.py（BGE-M3 向量化, 1024 維）
        ▼
  PostgreSQL + pgvector（db/schema.sql）

線上（查詢）
  使用者問題
        │  ingest/embedder.py（向量化）
        ▼
  query/retriever.py（向量檢索 + pg_trgm 關鍵字檢索 → RRF 混合排序）
        │
        ▼
  query/reranker.py（選用：cross-encoder 重排序，取 Top-K）
        │
        ▼
  query/generator.py（組 prompt → 呼叫 Gemini → 附加免責聲明）
        │
        ▼
  query/qa_engine.py（統一入口 answer_question()）
        │
        ├── app.py         CLI 介面
        └── api.py          FastAPI /ask 端點
```

## 技術選型

| 項目 | 選用 |
|---|---|
| 向量化模型 | `BAAI/bge-m3`（透過 `sentence-transformers` 載入，1024 維 dense 向量）|
| 向量資料庫 | PostgreSQL + `pgvector`（HNSW 索引，cosine 距離）|
| 關鍵字檢索 | `pg_trgm`（字元 trigram，中文免額外斷詞套件）|
| 混合檢索融合 | Reciprocal Rank Fusion (RRF) |
| 重排序（選用） | `BAAI/bge-reranker-v2-m3`（`sentence-transformers` 的 `CrossEncoder`）|
| 生成式模型 | Google Gemini，透過官方 `google-genai` SDK |

> **關於 Gemini SDK**：`google-generativeai` 套件已由 Google 官方棄用，統一改為
> `google-genai`（`from google import genai`）。本專案採用新版 SDK。模型名稱
> （如 `gemini-flash-latest`）異動頻率較高，建議上線前至
> [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)
> 確認目前可用的模型代號，並更新 `.env` 中的 `GEMINI_MODEL`。

## 專案結構

```
legal-rag-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── config.py                   # 集中管理所有可調參數
├── docker-compose.yml          # 一鍵啟動 PostgreSQL + pgvector
├── app.py                      # CLI 入口
├── api.py                      # FastAPI 入口（/ask）
├── db/
│   ├── schema.sql              # 資料表 + HNSW/trgm 索引
│   └── db_utils.py             # 連線輔助工具
├── ingest/                     # 離線資料前處理
│   ├── fetch_law.py            # 從官方開放資料下載並解析刑法全文
│   ├── chunker.py              # 智慧分塊（依條 + 長條文 overlap 切分）
│   ├── embedder.py             # BGE-M3 向量化
│   └── ingest_pipeline.py      # 串接以上三者、寫入資料庫（主要執行入口）
├── query/                      # 線上查詢
│   ├── retriever.py            # 向量檢索 / 關鍵字檢索 / 混合檢索(RRF)
│   ├── reranker.py             # cross-encoder 重排序
│   ├── generator.py            # Prompt 組裝 + 呼叫 Gemini + 免責聲明
│   └── qa_engine.py            # 對外主入口 answer_question()
├── data/
│   └── sample_criminal_code.json  # 離線測試用的少量示範條文（非最新版本）
└── examples/
    └── example_usage.py        # 直接呼叫函式 / 呼叫 API 的範例
```

## 快速開始

### 1. 安裝相依套件

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `sentence-transformers` 會連帶安裝 PyTorch，首次安裝體積較大；若有 GPU，
> 安裝好對應版本的 PyTorch 後，將 `.env` 中的 `EMBEDDING_DEVICE` /
> `RERANKER_DEVICE` 改成 `cuda` 可大幅加速。

### 2. 啟動 PostgreSQL（pgvector）

最簡單的方式是用 Docker：

```bash
docker compose up -d
```

或使用你自己既有的 PostgreSQL，但需先手動安裝 `pgvector` 擴充套件
（`CREATE EXTENSION vector;`，`pg_trgm` 為 PostgreSQL 內建 contrib 套件）。

### 3. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env，至少要填入 DATABASE_URL 與 GEMINI_API_KEY
```

Gemini API 金鑰可在 [Google AI Studio](https://aistudio.google.com/app/apikey) 免費申請。

### 4. 建表 + 匯入刑法條文（離線流程）

```bash
# 正式使用：從法務部「政府資料開放平臺」官方資料抓取最新全文
python -m ingest.ingest_pipeline

# 若暫時沒有網路，可先用內建的少量示範條文快速測試整條 pipeline
python -m ingest.ingest_pipeline --use-sample
```

這個指令會依序：建立/更新資料表 → 下載並解析刑法全文 → 依條分塊 → 用 BGE-M3
向量化 → 寫入 PostgreSQL。寫入時**不是**單純的 `ON CONFLICT` upsert，而是「整
部法規全刪再重建」：先刪除該法規（`law_pcode`）底下所有舊資料，再整批寫入這
次解析出的最新條文與分塊，兩步驟包在同一個交易裡。這是為了因應立法院修法──
詳見〈[修法後如何更新資料](#修法後如何更新資料)〉。可安全重複執行（例如排程
cron 定期重跑）。

### 5. 開始查詢

```bash
# CLI 互動模式
python app.py

# 或啟動 HTTP API
uvicorn api:app --reload --port 8000
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "我朋友欠我錢不還，我趁他不注意把他放在桌上的手機拿走抵債，這樣算什麼罪？"}'

# 或直接跑範例腳本
python examples/example_usage.py
```

---

## 資料來源說明（重要）

原始需求提到的網址
`https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=C0000001` 是全國法規資料庫
的**網頁瀏覽頁面**，其 `robots.txt` **明確禁止自動化程式存取**，因此本專案
**沒有**直接對該頁面寫爬蟲。

改用法務部透過「政府資料開放平臺」正式提供、允許重製利用的官方資料檔（政府
資料開放授權條款–第1版，免費）：

- 資料集頁面：<https://data.gov.tw/dataset/18289>（中文法規_法律資料檔下載）
- 檔案下載（RAW XML，每月更新）：
  `https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=RAW_XML&AuData=CF`

`ingest/fetch_law.py` 會下載這份官方 XML（內含「法律」層級的全部法規），從中
找出 `LawName` 等於「中華民國刑法」的節點並解析出條文清單、章節路徑、法規
最新異動日期（作為版本標示）與官方來源網址，一併存入資料庫供查詢時附上引用
來源。

**請注意**：由於官方頁面禁止自動化存取，我們在開發階段**無法即時撈取真實
XML 內容來 100% 驗證欄位標籤**（`LawName` / `ArticleType` / `ArticleNo` /
`ArticleContent` 等）目前是否仍是這個命名方式。`fetch_law.py` 已寫成對多種
候選標籤名稱寬容的邏輯，並提供除錯指令：

```bash
python -m ingest.fetch_law --inspect
```

此指令只會下載並印出 XML 結構，**不會**寫入任何資料，方便你在正式匯入前
人工核對條文數量、內容是否正確、是否為最新版本。若欄位名稱與程式假設不同，
只需調整 `ingest/fetch_law.py` 檔案開頭的 `_LAW_NAME_TAGS` 等候選清單常數。

如果你的環境無法對外連線至 `sendlaw.moj.gov.tw`，也可以：

1. 手動從 <https://data.gov.tw/dataset/18289> 下載 XML/ZIP，放到
   `data/law_bulk_raw.xml` 後直接執行
   `python -m ingest.ingest_pipeline --skip-schema`（會優先讀取本機快取）；
2. 或參考 `data/sample_criminal_code.json` 的格式，手動整理你信任來源的條文
   成同樣的 JSON 結構，再用 `ingest.chunker.chunk_law` /
   `ingest.ingest_pipeline.upsert_chunks` 串接後續流程。

## 修法後如何更新資料

《中華民國刑法》幾乎每年都會修訂幾條（法務部官方 XML 的〈沿革內容〉光是近幾
年就修了好幾次），因此本專案假設**資料匯入流程一定會被重複執行**，而不是只
跑一次性的匯入。

排程重跑的方式很單純：

```bash
python -m ingest.ingest_pipeline --force-download
```

跑一次這個指令，內部依序會發生：

1. `ingest/fetch_law.py` 重新下載官方 XML，解析出修法後的最新條文。
2. `ingest/chunker.py` 重新分塊——條文字數若因修法變多或變少，切出來的 chunk
   數量會跟著變（例如某條原本落落長被切成 3 塊，修完只剩 1 塊）。
3. `ingest/ingest_pipeline.py` 的 `upsert_chunks()` 把新的 chunk 寫回資料庫。

第 3 步是關鍵。單純用 `ON CONFLICT (law_pcode, article_no, chunk_index)
DO UPDATE` 覆蓋資料是不夠的：它只會更新「`chunk_index` 對得上」的舊列。如果
某條文修法後變短，多出來的舊 `chunk_index`（如上例的 1、2）不會被任何機制
刪除，就會變成內容仍是修法前舊條文的孤兒資料，之後向量檢索有機率把它們也
撈進來，等於系統偷偷用舊法條回答使用者，而且不容易發現。

因此 `upsert_chunks()` 改成「整部法規全刪再重建」，並包在同一個資料庫交易
裡（見 `ingest/ingest_pipeline.py`，對應的限制說明也記在 `db/schema.sql` 的
`law_chunks` 資料表註解中）：

1. 先 `DELETE FROM law_chunks WHERE law_pcode = ...`，清空這部法規底下所有
   舊資料。
2. 再把這次解析出的全部最新 chunk 重新 `INSERT`。
3. 兩步驟到最後才 `commit`；任一步失敗都會整個 `rollback`，不會留下「舊資料
   刪了、但新資料還沒寫完」的半殘狀態。由於包在同一個交易裡，PostgreSQL 的
   MVCC 也保證匯入進行中其他連線查詢仍看得到（尚未 commit 前的）舊資料，
   不會有使用者查詢時看到「表格是空的」這種情況。

結論：不論條文變長、變短，或整條被刪掉、新增，`--force-download` 都可以
安全地重複執行，資料庫裡永遠只留最新版本，不會有新舊版本混雜。建議排程
（如 cron）定期重跑，詳見〈[部署建議](#部署建議)〉。

## 進階功能：混合檢索與重排序

- **混合檢索（Hybrid Search）**：`query/retriever.py` 同時做向量檢索（語意）
  與 `pg_trgm` 關鍵字檢索（字面），再用 Reciprocal Rank Fusion 融合排名。
  好處是即使使用者用詞與條文原文差異很大（語意檢索強項），或反過來用詞很
  精確、剛好命中條文用字（關鍵字檢索強項），都有機會被檢索到。可用
  `ENABLE_HYBRID_SEARCH=false` 關閉，退回純向量檢索。
- **重排序（Re-ranking）**：`query/reranker.py` 用 cross-encoder
  （`BAAI/bge-reranker-v2-m3`）對第一階段選出的候選（`TOP_K_CANDIDATES`，
  預設 20 筆）重新評分，取最相關的 `TOP_K_FINAL`（預設 5 筆）送進 LLM，
  通常能提升最終送入 LLM 的條文品質。可用 `ENABLE_RERANK=false` 關閉以加快
  查詢速度。

## 使用範例

```python
from query.qa_engine import answer_question

result = answer_question("我拿刀威脅便利商店店員交出收銀機裡的錢，這樣算什麼罪？")
print(result.answer)          # 已包含免責聲明
for src in result.sources:
    print(src.law_name, src.article_no, src.source_url)
```

更多範例請見 `examples/example_usage.py`。

## 部署建議

- **開發/個人使用**：`docker compose up -d` 啟動資料庫 + `python app.py` 即可。
- **小型服務**：`uvicorn api:app --host 0.0.0.0 --port 8000`，建議在前面加
  nginx 或雲端服務商的負載平衡器，並限制 API 呼叫頻率（Gemini API 有其自身
  的用量限制與費用）。
- **模型執行位置**：`sentence-transformers` 的向量化與重排序模型會在你執行
  程式的機器上本地推論（非呼叫外部 API），若要壓低查詢延遲，建議部署在有
  GPU 的機器上，並把 `EMBEDDING_DEVICE` / `RERANKER_DEVICE` 設為 `cuda`；
  純 CPU 也可運作，但單次查詢的重排序步驟會較慢。
- **資料更新**：官方資料每月更新一次，且立法院幾乎每年都會修法，建議排程
  （如 cron）定期重跑 `python -m ingest.ingest_pipeline --force-download`。
  這個指令對「修法後條文變長、變短」是安全、可重複執行的（見〈[修法後如何
  更新資料](#修法後如何更新資料)〉），資料庫不會殘留修法前的舊條文。
- **機密資訊**：`.env` 內含 API 金鑰，切勿提交到版控或公開分享；
  `docker-compose.yml` 中的資料庫密碼僅為本地開發範例，正式環境請務必更換。

## 法律免責聲明

`query/generator.py` 中的 `DISCLAIMER` 常數會**強制附加**在每一次回答的最後，
不依賴 LLM 是否遵守 system instruction。系統提示詞（`SYSTEM_INSTRUCTION`）也
明確要求模型：

- 只能根據檢索到的參考條文作答，不可杜撰條文或刑度；
- 資訊不足時要明講、並說明還缺少哪些關鍵事實；
- 不得提供規避法律追訴的操作建議。

**但即使有這些機制，本系統仍然只是輔助工具**：RAG 檢索可能漏掉相關條文、
LLM 的分析可能有誤、且真實案件的定罪與量刑需要法院綜合全部事實、證據與
法律適用來判斷，非單一 AI 系統所能取代。如有實際法律需求，請洽詢律師或
法律扶助基金會。

## 已知限制與後續優化方向

- 目前只索引「法條本文」，未涵蓋大法官解釋、判決先例、學說見解，這些對於
  精確的法律判斷往往同樣重要，可考慮後續擴充資料來源（如司法院裁判書系統）。
- `pg_trgm` 屬字元層級比對，並非真正的中文斷詞全文檢索；若追求更精準的關鍵字
  檢索，可評估安裝 `zhparser` 等中文全文檢索擴充套件並改寫
  `query/retriever.py` 的 `keyword_search`。
- 章節路徑（`chapter_path`）的解析邏輯是用「編/章/節/款/目」關鍵字做簡易層級
  判斷，遇到罕見的條文結構命名可能不夠精確，僅作為輔助脈絡使用。
- 未實作對話歷史（multi-turn），目前每次查詢都是獨立的單輪問答；如需支援
  「這個問題我剛剛有提到的情況延伸問…」，需額外設計對話狀態管理。
- 條文修正後，法定刑度、要件都可能改變，務必定期重跑資料匯入流程以確保
  資料庫內容與官方最新公告一致；重跑本身已對 chunk 數量變化做防呆（見
  〈[修法後如何更新資料](#修法後如何更新資料)〉），不用擔心舊條文殘留。
