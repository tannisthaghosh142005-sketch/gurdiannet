FROM python:3.11-slim

WORKDIR /app

# Install system deps for opencv (headless) and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run inference script (evaluation mode)
# Override with: docker run ... streamlit run app.py --server.port 7860
CMD ["python", "inference.py"]