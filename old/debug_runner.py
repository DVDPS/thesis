#!/usr/bin/env python
"""
Debug runner for identifying import issues in the 2048 RL training system.
"""

import os
import sys
import importlib
import traceback

def check_import(module_path):
    """Try to import a module and print detailed error info if it fails"""
    print(f"Trying to import {module_path}...")
    try:
        module = importlib.import_module(module_path)
        print(f"✓ Successfully imported {module_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {module_path}: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return False

def main():
    # Add src to Python path
    sys.path.insert(0, os.path.abspath('src'))
    print(f"Python path: {sys.path}")
    
    # Check if we can import the key modules
    modules = [
        "thesis",
        "thesis.config",
        "thesis.environment.game2048",
        "thesis.environment.improved_reward",
        "thesis.agents.base_agent",
        "thesis.agents.enhanced_agent",
        "thesis.utils.curriculum_learning",
        "thesis.utils.enhanced_exploration",
        "thesis.utils.visualizations",
        "thesis.training.training",
        "thesis.training.simplified_training",
        "thesis.main"
    ]
    
    success = True
    for module in modules:
        if not check_import(module):
            success = False
    
    if success:
        print("\n✓ All modules imported successfully!")
        print("Trying to run the main module...")
        try:
            from thesis.main import main
            main()
            print("✓ Main module executed successfully!")
        except Exception as e:
            print(f"✗ Error running main module: {e}")
            print("\nTraceback:")
            traceback.print_exc()
    else:
        print("\n✗ Some modules failed to import. Fix the issues above and try again.")

if __name__ == "__main__":
    main() 