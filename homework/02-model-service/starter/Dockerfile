FROM python:3.13.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/models \
    LOG_PATH=/logs/requests.jsonl

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY model.py service.py ./
COPY proto/ ./proto/
RUN python -m grpc_tools.protoc -Iproto \
        --python_out=. --grpc_python_out=. proto/predict.proto

EXPOSE 8000 50051
ENTRYPOINT ["python", "service.py"]