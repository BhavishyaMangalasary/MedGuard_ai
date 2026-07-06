# Dockerfile
# MedGuard FastAPI web service
#
# Build:  docker build -t medguard .
# Run:    docker run -p 8080:8080 \
#           -e GOOGLE_API_KEY=your_key \
#           -e MEDGUARD_ENCRYPTION_KEY=your_key \
#           -e GOOGLE_GENAI_USE_VERTEXAI=FALSE \
#           medguard

FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer -- only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY services/ ./services/
COPY api_server.py .
COPY demo.py .

# Create data directory for encrypted patient records
RUN mkdir -p data/patients

# Expose the FastAPI port
EXPOSE 8080

# Health check -- Cloud Run uses this to verify the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/healthz || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]