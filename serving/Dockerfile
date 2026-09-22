# CPU-only serving image for the two-tower retrieval service.
# Build from the repo root:  docker build -f serving/Dockerfile -t recsys-tower .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Serving + base requirements. The requirements file points pip at PyTorch's
# CPU wheel index, so this stays a CPU image (no CUDA).
COPY requirements.txt serving/requirements.txt ./
COPY serving/requirements.txt ./serving-requirements.txt
RUN pip install -r requirements.txt -r serving-requirements.txt

# App code (artifacts/ are NOT baked in -- mount them or copy them in):
COPY models.py dataset.py ./
COPY serving/ ./serving/

# Required at runtime: artifacts/twotower.pt + artifacts/dataset.pt.
# Train them locally (python train.py) and bind-mount, e.g.:
#   docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts recsys-tower

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
