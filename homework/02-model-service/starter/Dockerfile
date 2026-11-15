FROM python:3.13.12-slim
WORKDIR /app

# Установите зависимости для CPU. Скопируйте model.py, service.py и proto/.
# Сгенерируйте классы Protobuf при сборке: в чистом клоне этих файлов нет.

# Настройте запуск python service.py в режиме http или grpc.
# Веса будут подключены в /models, а общий журнал — в /logs/requests.jsonl.
# Настройки для работы без сети приведены в шаге 2 задания.
