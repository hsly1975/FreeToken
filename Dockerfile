# FreeToken Docker image
#
# Based on community PR FlashML-org/FreeToken#295 (recrudesce), adapted:
#   - installs FreeToken FROM SOURCE (git main) instead of the PyPI release,
#     so the image always carries the latest qwen3_5_moe / QuantConfig fixes
#     (e.g. PR #438, which the 0.1.2 PyPI wheel predates)
#   - pins torch to the range required by pyproject.toml (>=2.11,<2.12, cu130)
#
# Build:  docker build -t freetoken:latest .
# Run:    see docker-compose.yml or docs/docker.md

# CUDA devel base so nvcc + headers are available for kernel compilation.
# The toolchain check only compares CUDA *major* versions (13 == 13), so a
# 13.3.1 toolkit is compatible with the cu130 torch wheel.
FROM nvidia/cuda:13.3.1-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
    CPATH="/usr/local/cuda/include:$CPATH" \
    LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"

# Build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv binary (official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Python 3.11 venv (avoids newer-CPython ABI issues with torch extensions).
# torch is pinned to the range pyproject.toml requires; the PyPI torch 2.11.x
# wheel is the cu130 build, so no extra index is needed.
RUN uv venv /opt/venv --python 3.11 && \
    uv pip install --no-cache "setuptools>=77" "torch>=2.11,<2.12" wheel ninja

# Install FreeToken from source so the image tracks git main (incl. #438).
# The C++ extensions (pinned_tensor, cpu_moe, ple_store) are built here;
# Triton/CUDA kernels are still JIT-compiled on first use.
COPY . /app
RUN uv pip install --no-cache ".[accel]"

EXPOSE 1919

ENTRYPOINT ["ft", "serve"]
CMD ["--host", "0.0.0.0", "--port", "1919"]
