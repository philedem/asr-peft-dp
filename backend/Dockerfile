# Use NVIDIA NGC PyTorch container optimized for GB10/Blackwell
# Recommended by NVIDIA: https://build.nvidia.com/spark/pytorch-fine-tune/instructions
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Install system dependencies (ffmpeg for audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install ML dependencies following NVIDIA's playbook recommendations
# Using specific versions known to work with GB10
RUN pip install --no-cache-dir \
    "torchao>=0.16.0" \
    transformers \
    peft \
    datasets \
    "trl==0.19.1" \
    "bitsandbytes==0.48"

# Copy requirements and install remaining app dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV HF_HOME=/workspace/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
