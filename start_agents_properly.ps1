#!/usr/bin/env powershell
# Reliable Multi-Agent Startup Script
# Properly starts each agent with correct module paths and dependency checks

param(
    [switch]$Background = $true,
    [switch]$ShowWindows = $false
)

# Load environment variables from .env file if it exists
if (Test-Path ".env") {
    Write-Host "Loading environment variables from .env file..." -ForegroundColor Green
    Get-Content .env | foreach {
        $name, $value = $_.split('=', 2)
        if ($name -and $value) {
            Set-Item -path env:$name -value $value
        }
    }
}

# Check if Google API key is set
if (-not $env:GOOGLE_API_KEY) {
    Write-Host "WARNING: GOOGLE_API_KEY not set. Gemini AI features will be disabled." -ForegroundColor Yellow
    Write-Host "Set your Google API key: `$env:GOOGLE_API_KEY = 'your_key_here'" -ForegroundColor Yellow
}

Write-Host "Starting Multi-Agent Framework with Proper Module Paths" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# Create logs directory if it doesn't exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "Created logs directory" -ForegroundColor Green
}

# Define agents with their correct paths and module names
$agents = @(
    @{
        Name = "Data Loader"
        Directory = "data-loader-agent"
        Module = "data_loader"
        Port = 10006
        Features = "CSV, JSON, Hyper API Support"
    },
    @{
        Name = "Data Analyst"
        Directory = "data-analyst-agent"
        Module = "data_analyst"
        Port = 10007
        Features = "Business Intelligence Analysis"
    },
    @{
        Name = "Data Cleaning"
        Directory = "data-cleaning-agent"
        Module = "data_cleaning_agent"
        Port = 10008
        Features = "Data Quality & Cleaning"
    },
    @{
        Name = "Data Enrichment"
        Directory = "data-enrichment-agent"
        Module = "data_enrichment_agent"
        Port = 10009
        Features = "Data Enhancement & Augmentation"
    },
    @{
        Name = "Presentation"
        Directory = "presentation-agent"
        Module = "presentation_agent"
        Port = 10010
        Features = "Report Generation & Visualization"
    },
    @{
        Name = "RootCause Analyst"
        Directory = "rootcause-analyst-agent"
        Module = "rootcause_analyst"
        Port = 10011
        Features = "Why-Bot AI Investigation"
    },
    @{
        Name = "Schema Profiler"
        Directory = "schema-profiler-agent"
        Module = "schema_profiler"
        Port = 10012
        Features = "AI Dataset Profiling & Caching"
    },
    @{
        Name = "Orchestrator"
        Directory = "orchestrator-agent"
        Module = "orchestrator_agent"
        Port = 10000
        Features = "Pipeline Coordination & Management"
    }
)

# Function to check if an agent is already running
function Test-AgentRunning {
    param($Port)
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

# Function to start an individual agent
function Start-Agent {
    param($Agent)
    
    Write-Host ""
    Write-Host "Starting $($Agent.Name) Agent..." -ForegroundColor Yellow
    Write-Host "   Directory: $($Agent.Directory)" -ForegroundColor Gray
    Write-Host "   Module: $($Agent.Module)" -ForegroundColor Gray
    Write-Host "   Port: $($Agent.Port)" -ForegroundColor Gray
    Write-Host "   Features: $($Agent.Features)" -ForegroundColor Gray
    
    # Check if already running
    if (Test-AgentRunning -Port $Agent.Port) {
        Write-Host "   Already running!" -ForegroundColor Green
        return $true
    }
    
    # Check if directory exists
    if (-not (Test-Path $Agent.Directory)) {
        Write-Host "   ERROR: Directory not found: $($Agent.Directory)" -ForegroundColor Red
        return $false
    }
    
    # Start the agent
    try {
        Push-Location $Agent.Directory
        
        $logFile = "../logs/$($Agent.Directory)_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
        
        if ($Background) {
            # Start in background
            $process = Start-Process python -ArgumentList "-m", $Agent.Module -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru -WindowStyle Hidden
            Write-Host "   Started in background (PID: $($process.Id))" -ForegroundColor Green
        } else {
            # Start in new window
            $windowStyle = if ($ShowWindows) { "Normal" } else { "Minimized" }
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m $($Agent.Module)" -WindowStyle $windowStyle
            Write-Host "   Started in new window" -ForegroundColor Green
        }
        
        # Wait a moment for startup
        Start-Sleep -Seconds 3
        
        # Verify it started
        if (Test-AgentRunning -Port $Agent.Port) {
            Write-Host "   SUCCESS: Verified running on port $($Agent.Port)" -ForegroundColor Green
            return $true
        } else {
            Write-Host "   WARNING: Started but not yet responding on port $($Agent.Port)" -ForegroundColor Yellow
            return $true
        }
    }
    catch {
        Write-Host "   ERROR: Failed to start: $_" -ForegroundColor Red
        return $false
    }
    finally {
        Pop-Location
    }
}

# Start all agents
$successCount = 0
foreach ($agent in $agents) {
    if (Start-Agent -Agent $agent) {
        $successCount++
    }
}

Write-Host ""
Write-Host "Startup Summary" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "Successfully started: $successCount/$($agents.Count) agents" -ForegroundColor Green

if ($successCount -eq $agents.Count) {
    Write-Host "All agents started successfully!" -ForegroundColor Green
} else {
    Write-Host "Some agents failed to start. Check logs in ./logs/ directory" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Agent Endpoints:" -ForegroundColor Cyan
foreach ($agent in $agents) {
    $status = if (Test-AgentRunning -Port $agent.Port) { "RUNNING" } else { "NOT RUNNING" }
    Write-Host "   $($agent.Name): http://localhost:$($agent.Port) - $status" -ForegroundColor White
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Wait 30 seconds for all agents to fully initialize" -ForegroundColor White
Write-Host "   2. Check status: python scripts/framework_cli.py --check" -ForegroundColor White
Write-Host "   3. Launch dashboard: python scripts/run_dashboard.py" -ForegroundColor White
Write-Host "   4. Run analysis: python scripts/framework_cli.py data/your_file.csv" -ForegroundColor White
Write-Host "   5. View logs: Get-ChildItem logs/*.log" -ForegroundColor White

Write-Host ""
Write-Host "Multi-Agent Framework Ready!" -ForegroundColor Green 