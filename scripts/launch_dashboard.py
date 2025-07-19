#!/usr/bin/env python3
"""
Robust Streamlit Dashboard Launcher
Handles path configuration and dependency checking automatically
"""

import sys
import os
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the Python environment for running the dashboard."""
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Set working directory to project root
    os.chdir(project_root)
    
    print(f"✅ Project root: {project_root}")
    print(f"✅ Working directory: {os.getcwd()}")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'httpx',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            *missing_packages
        ], check=True)
        print("✅ All packages installed successfully")
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n🚀 Launching Streamlit Dashboard...")
    
    dashboard_path = Path("scripts/streamlit_dashboard.py")
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
        return True
    
    return True

def main():
    """Main launcher function."""
    print("🤖 Multi-Agent Data Analysis Dashboard Launcher")
    print("=" * 50)
    
    try:
        # Setup environment
        setup_environment()
        
        # Check dependencies
        check_dependencies()
        
        # Launch dashboard
        launch_dashboard()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 