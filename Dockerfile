FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Run inference, then start a simple HTTP server to keep the container alive
CMD python inference.py; python server/app.py
