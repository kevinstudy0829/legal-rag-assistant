# 刑法查詢助手 API（FastAPI + uvicorn）
#
# 注意：sentence-transformers 會連帶安裝 PyTorch，image 會比較大（GB 等級），
# 首次 build 需要一段時間；建好之後除非 requirements.txt 改變，重建會走快取。
FROM python:3.12-slim

WORKDIR /app

# psycopg2-binary 已內含編譯好的 libpq，但執行期仍需要 libpq5 動態函式庫
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
