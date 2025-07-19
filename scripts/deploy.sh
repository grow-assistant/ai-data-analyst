#!/bin/bash

# AI Data Analyst Multi-Agent Framework Deployment Script
# Copyright 2025 Google LLC

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
ENV_FILE="${PROJECT_ROOT}/.env"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    success "All dependencies are installed."
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/outputs"
    mkdir -p "${PROJECT_ROOT}/sessions"
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/monitoring/grafana/provisioning"
    
    success "Directories created."
}

# Setup environment file
setup_environment() {
    log "Setting up environment configuration..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# AI Data Analyst Multi-Agent Framework Environment
COMPOSE_PROJECT_NAME=ai-data-analyst
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Monitoring
PROMETHEUS_RETENTION=200h
GRAFANA_ADMIN_PASSWORD=admin123

# Security
API_KEY_SECRET=your-secret-key-here

# Performance
WORKER_PROCESSES=1
MAX_REQUESTS=1000
EOF
        success "Environment file created at $ENV_FILE"
        warning "Please review and update the environment variables in $ENV_FILE"
    else
        log "Environment file already exists."
    fi
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build with build args for caching
    docker-compose build --parallel --progress=plain
    
    success "Docker images built successfully."
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Pull external images first
    docker-compose pull prometheus grafana otel-collector || true
    
    # Start services in dependency order
    log "Starting monitoring services..."
    docker-compose up -d prometheus grafana otel-collector
    
    sleep 10
    
    log "Starting agent services..."
    docker-compose up -d
    
    success "Services deployed successfully."
}

# Health check
health_check() {
    log "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    # List of services to check
    services=(
        "data-loader:10006"
        "data-cleaning:10008"
        "data-cleaning-tools:11008"
        "data-enrichment:10009"
        "data-analyst:10007"
        "rootcause-analyst:10011"
        "presentation:10010"
        "schema-profiler:10012"
        "orchestrator:10005"
    )
    
    for service in "${services[@]}"; do
        local service_name="${service%:*}"
        local port="${service#*:}"
        
        log "Checking $service_name health..."
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
                success "$service_name is healthy"
                break
            else
                if [[ $attempt -eq $max_attempts ]]; then
                    error "$service_name health check failed after $max_attempts attempts"
                    return 1
                fi
                log "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
                sleep 10
                ((attempt++))
            fi
        done
        attempt=1
    done
    
    success "All services are healthy!"
}

# Display status
show_status() {
    log "Service Status:"
    docker-compose ps
    
    echo ""
    log "Access URLs:"
    echo "🎯 Orchestrator API: http://localhost:10005"
    echo "📊 Grafana Dashboard: http://localhost:3000 (admin/admin123)"
    echo "📈 Prometheus: http://localhost:9090"
    echo "🔍 OpenTelemetry Collector: http://localhost:8888/metrics"
    
    echo ""
    log "Individual Agent URLs:"
    echo "📁 Data Loader: http://localhost:10006"
    echo "🧹 Data Cleaning: http://localhost:10008"
    echo "🛠️ Cleaning Tools (MCP): http://localhost:11008"
    echo "🔧 Data Enrichment: http://localhost:10009"
    echo "📊 Data Analyst: http://localhost:10007"
    echo "🔍 Root Cause Analyst: http://localhost:10011"
    echo "📋 Presentation: http://localhost:10010"
    echo "📝 Schema Profiler: http://localhost:10012"
}

# Stop services
stop_services() {
    log "Stopping services..."
    cd "$PROJECT_ROOT"
    docker-compose down
    success "Services stopped."
}

# Clean up
cleanup() {
    log "Cleaning up..."
    cd "$PROJECT_ROOT"
    
    # Stop and remove containers
    docker-compose down -v --remove-orphans
    
    # Remove images (optional)
    if [[ "${1:-}" == "--full" ]]; then
        log "Removing Docker images..."
        docker-compose down --rmi all
        docker system prune -f
    fi
    
    success "Cleanup completed."
}

# Show logs
show_logs() {
    local service="${1:-}"
    cd "$PROJECT_ROOT"
    
    if [[ -n "$service" ]]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Main function
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy"|"start")
            check_dependencies
            create_directories
            setup_environment
            build_images
            deploy_services
            health_check
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 5
            deploy_services
            health_check
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "health")
            health_check
            ;;
        "cleanup")
            cleanup "${2:-}"
            ;;
        "build")
            build_images
            ;;
        *)
            echo "Usage: $0 {deploy|start|stop|restart|status|logs [service]|health|cleanup [--full]|build}"
            echo ""
            echo "Commands:"
            echo "  deploy/start  - Build and deploy all services"
            echo "  stop          - Stop all services"
            echo "  restart       - Restart all services"
            echo "  status        - Show service status and URLs"
            echo "  logs [service] - Show logs (optionally for specific service)"
            echo "  health        - Run health checks"
            echo "  cleanup [--full] - Clean up containers and optionally images"
            echo "  build         - Build Docker images only"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 