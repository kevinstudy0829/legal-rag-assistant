-- ============================================================
-- 刑法查詢助手 - 資料庫 Schema
-- 使用方式： psql "$DATABASE_URL" -f db/schema.sql
-- ============================================================

-- 向量檢索
CREATE EXTENSION IF NOT EXISTS vector;
-- 關鍵字（trigram）檢索，用於混合檢索；對中文以字元級 n-gram 比對，
-- 不需額外安裝中文斷詞套件（如 zhparser）即可運作
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS law_chunks (
    id              BIGSERIAL PRIMARY KEY,

    -- 法規層級的中繼資料
    law_name        TEXT NOT NULL,              -- 例如「中華民國刑法」
    law_pcode       TEXT NOT NULL,              -- 法務部法規代碼，例如 C0000001
                                                  -- 注意：UNIQUE 約束用到這個欄位，NULL 會讓 upsert 去重失效
                                                  -- （NULL 在 UNIQUE/ON CONFLICT 判斷中永遠視為互不相等），故設為 NOT NULL
    law_modified_date TEXT,                     -- 法規最新異動日期（用來標示版本）
    source_url      TEXT,                       -- 官方條文來源網址

    -- 條文層級的中繼資料
    article_no      TEXT NOT NULL,              -- 例如「第 271 條」
    article_no_sort INT,                        -- 條號排序用的數字（例如 271），之1之2另外處理
    chapter_path    TEXT,                       -- 章節路徑，例如「第二編 分則 / 第二十二章 殺人罪」

    -- 內容與分塊
    content         TEXT NOT NULL,              -- 該分塊的條文內容（含條號前綴，利於獨立閱讀）
    chunk_index     INT NOT NULL DEFAULT 0,      -- 若同一條被拆成多塊，記錄順序（0 起算）
    chunk_count     INT NOT NULL DEFAULT 1,      -- 該條文總共被拆成幾塊

    -- 向量（BGE-M3 dense 向量固定 1024 維）
    embedding       vector(1024),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 同一部法規、同一條、同一分塊只會存在一筆，方便重新匯入時 UPSERT
    --
    -- ⚠️ 注意：UPSERT（ON CONFLICT ... DO UPDATE）只會覆蓋「chunk_index 對得上」的舊資料，
    -- 不會自動刪除多出來的舊 chunk。若某條文修法後變短、chunk 數變少（例如從 3 塊縮成 1 塊），
    -- 舊的 chunk_index=1、2 會變成孤兒資料，永久留在資料庫與向量索引中並可能被誤檢索到。
    -- 匯入端（ingest_pipeline.py）必須自行處理，做法二選一：
    --   (a) 每次匯入前，對每個 (law_pcode, article_no) 先 DELETE 掉 chunk_index >= 新 chunk_count 的舊列；
    --   (b) 或更保守：每次匯入前，直接把該 law_pcode 底下所有舊資料整批刪除再全部重新 INSERT。
    UNIQUE (law_pcode, article_no, chunk_index)
);

-- 向量近似最近鄰索引（HNSW，pgvector 0.5+ 支援；查詢時使用 cosine 距離 <=>）
CREATE INDEX IF NOT EXISTS idx_law_chunks_embedding_hnsw
    ON law_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 關鍵字（trigram）索引，供混合檢索使用
CREATE INDEX IF NOT EXISTS idx_law_chunks_content_trgm
    ON law_chunks USING gin (content gin_trgm_ops);

-- 條號查詢索引（使用者若直接問「第271條」可快速命中）
CREATE INDEX IF NOT EXISTS idx_law_chunks_article_no
    ON law_chunks (law_pcode, article_no);

CREATE INDEX IF NOT EXISTS idx_law_chunks_law_name
    ON law_chunks (law_name);

-- 自動維護 updated_at
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_law_chunks_updated_at ON law_chunks;
CREATE TRIGGER trg_law_chunks_updated_at
    BEFORE UPDATE ON law_chunks
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
