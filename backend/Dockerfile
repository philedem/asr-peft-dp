# Multi-stage build: builder with CUDA devel tools, production with runtime only
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        cmake \
        git \
        libopenblas-dev \
        libgomp1 \
        ffmpeg \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /build

# Install PyTorch with CUDA support
RUN pip3 install --break-system-packages --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Clone and build CTranslate2 from source with CUDA support for newer architectures
RUN echo "=== Cloning CTranslate2 v4.5.0 ===" \
    && git clone --depth=1 --branch v4.5.0 --recurse-submodules --shallow-submodules \
       https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2

# Configure CTranslate2 with CUDA support including sm_89 and sm_90 (closest to sm_121)
RUN echo "=== Configuring CTranslate2 with CMake ===" \
    && cd /tmp/ctranslate2 \
    && mkdir build && cd build \
    && cmake .. \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCUDNN_ROOT=/usr \
        -DCMAKE_CUDA_ARCHITECTURES="89;90" \
        -DOPENMP_RUNTIME=COMP \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=ON

# Build CTranslate2 (this takes time)
RUN echo "=== Building CTranslate2 ===" \
    && cd /tmp/ctranslate2/build \
    && cmake --build . --parallel $(nproc) \
    && cmake --install .

# Install pybind11 first (required by CTranslate2 Python bindings)
RUN pip3 install --break-system-packages pybind11 --no-cache-dir

# Install CTranslate2 Python bindings
RUN echo "=== Installing CTranslate2 Python bindings ===" \
    && cd /tmp/ctranslate2/python \
    && pip3 install --break-system-packages . --no-cache-dir --verbose --no-build-isolation

# Install other Python dependencies
COPY requirements.txt /build/
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Production stage: minimal runtime image
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        ffmpeg \
        libsndfile1 \
        libopenblas0 \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy CTranslate2 C++ library from builder
COPY --from=builder /usr/local/lib/libctranslate2.so* /usr/local/lib/
COPY --from=builder /usr/local/include/ctranslate2 /usr/local/include/ctranslate2

# Copy Python environment from builder (packages and binaries)
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Update library path
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Set HuggingFace cache directory
ENV HF_HOME=/workspace/.cache/huggingface

# Enable sm_120 compatibility for sm_121 (Blackwell GB10)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_NVFUSER_DISABLE=1
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Copy backend app
WORKDIR /workspace
COPY . .

EXPOSE 8000

CMD ["/usr/bin/python3.12", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]