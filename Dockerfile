# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (useful for FAISS, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY backend /app/
COPY configs /app/

# Expose port
EXPOSE 8000

# Start app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
