# Simple Multi-Agent Startup Script  
# Starts all agents with correct module paths

Write-Host "Starting Multi-Agent Framework" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Create logs directory if it doesn't exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "Created logs directory" -ForegroundColor Green
}

# Define agents with their correct paths and module names
$agents = @(
    @{Name="Data Loader"; Directory="data-loader-agent"; Module="data_loader"; Port=10006},
    @{Name="Data Analyst"; Directory="data-analyst-agent"; Module="data_analyst"; Port=10007},
    @{Name="Data Cleaning"; Directory="data-cleaning-agent"; Module="data_cleaning_agent"; Port=10008},
    @{Name="Data Enrichment"; Directory="data-enrichment-agent"; Module="data_enrichment_agent"; Port=10009},
    @{Name="Presentation"; Directory="presentation-agent"; Module="presentation_agent"; Port=10010},
    @{Name="RootCause Analyst"; Directory="rootcause-analyst-agent"; Module="rootcause_analyst"; Port=10011},
    @{Name="Schema Profiler"; Directory="schema-profiler-agent"; Module="schema_profiler"; Port=10012},
    @{Name="Orchestrator"; Directory="orchestrator-agent"; Module="orchestrator_agent"; Port=10000}
)

Write-Host ""
Write-Host "Starting agents..." -ForegroundColor Yellow

# Start each agent
foreach ($agent in $agents) {
    Write-Host "Starting $($agent.Name) on port $($agent.Port)..." -ForegroundColor White
    
    # Check if directory exists
    if (-not (Test-Path $agent.Directory)) {
        Write-Host "   ERROR: Directory not found: $($agent.Directory)" -ForegroundColor Red
        continue
    }
    
    # Start the agent in a new PowerShell window
    $command = "cd '$($agent.Directory)'; python -m $($agent.Module)"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $command -WindowStyle Minimized
    
    Write-Host "   Started in new window" -ForegroundColor Green
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "All agents started!" -ForegroundColor Green
Write-Host ""
Write-Host "Agent Endpoints:" -ForegroundColor Cyan
foreach ($agent in $agents) {
    Write-Host "   $($agent.Name): http://localhost:$($agent.Port)" -ForegroundColor White
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Wait 30 seconds for all agents to initialize" -ForegroundColor White
Write-Host "   2. Test health: Invoke-WebRequest http://localhost:10000/health" -ForegroundColor White  
Write-Host "   3. Launch dashboard: python scripts/run_dashboard.py" -ForegroundColor White
Write-Host "   4. Check status: python scripts/framework_cli.py --check" -ForegroundColor White

Write-Host ""
Write-Host "Framework Ready!" -ForegroundColor Green 