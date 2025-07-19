#!/usr/bin/env python3
"""
Launcher script for the Multi-Agent Data Analysis Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'httpx', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_packages
            ])
            print("✅ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Starting Multi-Agent Data Analysis Dashboard")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("data").exists():
        print("❌ Error: 'data' folder not found!")
        print("Please run this script from the agents root directory.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check if orchestrator might be running
    print("\n📋 Pre-flight checks:")
    print("1. Make sure the orchestrator agent is running (port 10000)")
    print("2. Start agents using: powershell ./start_all_agents.ps1")
    print("3. Wait for all agents to be healthy before using the dashboard")
    
    print("\n🌐 Starting Streamlit dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "scripts/streamlit_dashboard.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 