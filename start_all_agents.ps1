# Enhanced Multi-Agent Startup Script
# Starts all agents including the new RootCause Analyst (Why-Bot)

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

Write-Host "Starting Enhanced Multi-Agent Framework with Complete Capabilities" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# Array of agents with their details
$agents = @(
    @{Name="Data Loader"; Path="data-loader-agent"; Port=10006; Features="Hyper API Support"},
    @{Name="Data Analyst"; Path="data-analyst-agent"; Port=10007; Features="Enhanced Analysis"},
    @{Name="Data Cleaning"; Path="data-cleaning-agent"; Port=10008; Features="Data Cleaning"},
    @{Name="Data Enrichment"; Path="data-enrichment-agent"; Port=10009; Features="Data Enrichment"},
    @{Name="Presentation"; Path="presentation-agent"; Port=10010; Features="Gemini AI Reports"},
    @{Name="RootCause Analyst"; Path="rootcause-analyst-agent"; Port=10011; Features="Why-Bot AI"},
    @{Name="Schema Profiler"; Path="schema-profiler-agent"; Port=10012; Features="AI Dataset Profiling"},
    @{Name="Orchestrator"; Path="orchestrator-agent"; Port=10000; Features="Pipeline Coordination"}
)

# Start each agent
foreach ($agent in $agents) {
    Write-Host ""
    Write-Host "Starting $($agent.Name) Agent..." -ForegroundColor Yellow
    Write-Host "   Port: $($agent.Port)" -ForegroundColor Gray
    Write-Host "   Features: $($agent.Features)" -ForegroundColor Gray
    
    # Change to agent directory and start
    Push-Location $agent.Path
    
    try {
        # Start the agent in a new PowerShell window
        $startCommand = "python -m $($agent.Path.Replace('-', '_')) 2>&1 | Tee-Object -FilePath ../logs/$($agent.Path)_$((Get-Date).ToString('yyyyMMdd_HHmmss')).log"
        
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $startCommand -WindowStyle Minimized
        
        Write-Host "   Started successfully" -ForegroundColor Green
        
        # Brief pause to avoid overwhelming the system
        Start-Sleep -Seconds 2
    }
    catch {
        Write-Host "   Failed to start: $_" -ForegroundColor Red
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "All Enhanced Agents Started!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green

Write-Host ""
Write-Host "Enhanced Capabilities Available:" -ForegroundColor Cyan
Write-Host "   * Tableau Hyper API - Fast TDSX/Hyper file loading" -ForegroundColor White
Write-Host "   * Comprehensive Analysis - 10+ business intelligence modules" -ForegroundColor White  
Write-Host "   * RootCause Analyst (Why-Bot) - AI hypothesis generation and testing" -ForegroundColor White
Write-Host "   * Google Gemini AI - Executive insights and recommendations" -ForegroundColor White
Write-Host "   * Professional Reporting - Executive-ready presentations" -ForegroundColor White
Write-Host "   * Smart Escalation - Automatic low-confidence flagging" -ForegroundColor White

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Wait 30 seconds for all agents to fully initialize" -ForegroundColor White
Write-Host "   2. Check status: python tdsx_cli.py --check" -ForegroundColor White
Write-Host "   3. Run analysis: python tdsx_cli.py your_data_file.csv" -ForegroundColor White
Write-Host "   4. View logs in ./logs/ directory" -ForegroundColor White

Write-Host ""
Write-Host "Agent Endpoints:" -ForegroundColor Cyan
foreach ($agent in $agents) {
    Write-Host "   $($agent.Name): http://localhost:$($agent.Port)" -ForegroundColor White
}

Write-Host ""
Write-Host "Please wait 30 seconds for all agents to fully initialize..." -ForegroundColor Yellow

# Create logs directory if it doesn't exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host ""
Write-Host "Enhanced Multi-Agent Framework Ready!" -ForegroundColor Green 