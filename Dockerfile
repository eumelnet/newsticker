FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY static/ static/

EXPOSE 8000

USER nobody

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
