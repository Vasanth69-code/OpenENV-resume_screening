FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package to make openenv.yaml and __init__ files accessible
COPY . .

# Run the app from the server module
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
