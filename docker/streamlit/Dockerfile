FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements-serving.txt /app/requirements-serving.txt

RUN pip install --no-cache-dir \
    -r /app/requirements-serving.txt

COPY src /app/src
COPY app /app/app

EXPOSE 8501

CMD ["streamlit", "run", "/app/app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]