#!/usr/bin/env python
"""
Main launcher for the 2048 RL training system.
Provides a unified interface to train and evaluate models.
"""

import os
import sys
import argparse
import importlib
import subprocess

def install_package():
    """Install the package in development mode"""
    print("Installing the package in development mode...")
    try:
        # Try to check if pip is available
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        except subprocess.CalledProcessError:
            print("Error: pip is not available. Please install pip first.")
            print("See: https://pip.pypa.io/en/stable/installation/")
            return False
        
        # Install requirements first
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Install the package in development mode
        print("Installing the package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        
        print("Installation complete. Please run the script again.")
        return True
    except Exception as e:
        print(f"Error during installation: {e}")
        print("\nAlternative installation methods:")
        print("1. Install manually:")
        print("   pip install -r requirements.txt")
        print("   pip install -e .")
        print("2. Run using PYTHONPATH:")
        print("   set PYTHONPATH=src")
        print("   python -m thesis.main")
        return False

def run_direct():
    """Run the main module directly"""
    # Add src to Python path
    sys.path.insert(0, os.path.abspath('src'))
    
    # Import the main module
    try:
        from thesis.main import main
        main()
        return True
    except ImportError as e:
        print(f"Error importing thesis.main: {e}")
        print("Make sure src is in the Python path.")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="2048 RL Training System")
    parser.add_argument("--install", action="store_true", help="Install the package in development mode")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode to check imports")
    parser.add_argument("--list-modes", action="store_true", help="List available training modes and exit")
    
    # Pass through all remaining arguments to the main module
    args, remaining_args = parser.parse_known_args()
    
    # Handle install option
    if args.install:
        if install_package():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Handle debug option
    if args.debug:
        print("Running in debug mode. Checking imports...")
        subprocess.call([sys.executable, "debug_runner.py"])
        sys.exit(0)
    
    # Handle list-modes option
    if args.list_modes:
        print("Available training modes:")
        print("  standard   - Standard PPO training")
        print("  simplified - Simplified training with stable parameters")
        print("  enhanced   - Enhanced agent training")
        print("  balanced   - Balanced exploration training")
        print("\nExample usage:")
        print("  python run_2048.py --mode enhanced --epochs 1000 --batch-size 64")
        print("  python run_2048.py --mode evaluate --checkpoint checkpoints/best_model.pt --games 10")
        sys.exit(0)
    
    # Run the main module directly
    sys.argv = [sys.argv[0]] + remaining_args
    if not run_direct():
        sys.exit(1)

if __name__ == "__main__":
    main() 