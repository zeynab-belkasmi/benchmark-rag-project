FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/app/data/models

# CPU torch first (lighter image); GPU via docker-compose.gpu.yml (planned)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data

CMD ["uvicorn", "ragbench.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
