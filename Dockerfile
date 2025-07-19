# Multi-stage build for AI Data Analyst Multi-Agent Framework
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements files
COPY requirements*.txt ./
COPY common_utils/pyproject.toml ./common_utils/
COPY */pyproject.toml ./agents/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || true

# Install agent-specific dependencies
RUN for dir in data-loader-agent data-cleaning-agent data-enrichment-agent \
               data-analyst-agent presentation-agent orchestrator-agent \
               rootcause-analyst-agent schema-profiler-agent; do \
    if [ -f "${dir}/pyproject.toml" ]; then \
        pip install -e "./${dir}" || true; \
    fi; \
done

# Install common utilities
RUN pip install -e ./common_utils || true

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_NAME="" \
    AGENT_PORT="" \
    LOG_LEVEL="INFO"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy Python packages from base stage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs sessions data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${AGENT_PORT:-8000}/health || exit 1

# Expose port (will be overridden by specific agent)
EXPOSE 8000

# Default command (will be overridden by specific agent)
CMD ["python", "-m", "uvicorn", "--host", "0.0.0.0", "--port", "8000"] 