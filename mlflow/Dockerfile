FROM python:3.12-slim

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow \
    boto3 \
    pymysql \
    cryptography

# Create directory for MLflow artifacts and set permissions
RUN mkdir -p /mlflow/artifacts /mlflow/db && \
    chmod -R 777 /mlflow

WORKDIR /mlflow

# Expose MLflow UI port
EXPOSE 5000

# Start MLflow tracking server
# --backend-store-uri: Where to store run/experiment metadata (SQLite)
# --default-artifact-root: Where to store artifacts (models, plots, etc.)
# --host 0.0.0.0: Listen on all interfaces
# --allowed-hosts: Allow connections from backend pods and external access
CMD mkdir -p /mlflow/artifacts /mlflow/db && \
    chmod -R 777 /mlflow && \
    mlflow server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts 'asr-peft-mlflow' \
    --allowed-hosts 'asr-peft-mlflow:5000' \
    --allowed-hosts '192.168.3.155' \
    --allowed-hosts '192.168.3.155:5000' \
    --allowed-hosts 'localhost' \
    --allowed-hosts '127.0.0.1'
