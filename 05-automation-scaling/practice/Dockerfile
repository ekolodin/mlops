FROM python:3.13.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# TODO 1: скопируйте pipeline.py в /app.
# TODO 2: задайте ENTRYPOINT, запускающий python /app/pipeline.py.
# Аргументы (all, report, --output...) будем передавать в docker run.
