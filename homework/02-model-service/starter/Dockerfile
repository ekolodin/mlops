FROM python:3.13.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY model.py service.py ./
COPY proto/ proto/
RUN python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/predict.proto

ENV MODEL_DIR=/models \
    LOG_PATH=/logs/requests.jsonl \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "service.py"]
