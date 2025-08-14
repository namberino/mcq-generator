FROM python:3.11-slim

# set HF cache to /tmp for writable FS on Spaces
ENV HF_HOME=/tmp/huggingface
ENV TOKENIZERS_PARALLELISM=false

# install system packages needed by some python libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
# try to be robust to wheels/build issues
# RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r /app/requirements.txt || true
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app code
COPY . /app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
