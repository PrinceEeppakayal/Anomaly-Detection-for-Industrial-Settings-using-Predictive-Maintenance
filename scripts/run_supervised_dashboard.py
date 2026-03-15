"""
🚀 SUPERVISED LEARNING DASHBOARD LAUNCHER
==========================================

Launches the supervised learning dashboard that uses:
- Random Forest Classifier (200 trees)
- Multi-class classification (7 classes)
- Labeled training data from simulator
- Per-class metrics (precision, recall, F1)
- Confusion matrix visualization

Usage:
    python run_supervised_dashboard.py
"""

import subprocess
import sys
import os
import time

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'sklearn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Please install requirements first:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def launch_supervised_dashboard():
    """Launch the supervised learning Streamlit dashboard"""
    
    print("🤖 SUPERVISED LEARNING ANOMALY DETECTION DASHBOARD")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if dashboard file exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_file = os.path.join(base_dir, "streamlit_dashboard_supervised.py")
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Supervised dashboard file not found: {dashboard_file}")
        return
    
    print("\n📊 Dashboard Features:")
    print("   • Random Forest Classifier (200 trees)")
    print("   • Multi-class classification (7 classes)")
    print("   • Labeled training data from simulator")
    print("   • Per-class metrics (Precision, Recall, F1)")
    print("   • Confusion matrix visualization")
    print("   • Prediction confidence scores")
    print("   • Auto-refresh every 2 seconds")
    
    print("\n🎯 Anomaly Classes:")
    classes = ['normal', 'overheating', 'power_surge', 'signal_degradation',
               'network_overload', 'cooling_failure', 'equipment_failure']
    for i, cls in enumerate(classes, 1):
        print(f"   {i}. {cls.upper()}")
    
    print("\n🌐 Dashboard will open in your web browser")
    print("   URL: http://localhost:8502")
    print("   Press Ctrl+C to stop the dashboard")
    
    # Small delay
    time.sleep(2)
    
    try:
        # Launch Streamlit on port 8502 to avoid conflicts with port 8503 (integrated)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_file, 
            "--server.port", "8502",
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--theme.base", "light"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Supervised dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_supervised_dashboard()
