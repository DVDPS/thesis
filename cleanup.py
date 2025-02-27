#!/usr/bin/env python
"""
Cleanup script for the 2048 RL project.
This removes duplicate/obsolete files and other unnecessary artifacts.
"""

import os
import shutil
import glob

def safe_remove(path):
    """Safely remove a file or directory if it exists"""
    if os.path.isfile(path):
        print(f"Removing file: {path}")
        os.remove(path)
    elif os.path.isdir(path):
        print(f"Removing directory: {path}")
        shutil.rmtree(path)
    else:
        print(f"Path does not exist, skipping: {path}")

def find_files(patterns):
    """Find files matching any of the given patterns"""
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return files

def main():
    # List of old/duplicate files to remove
    obsolete_files = [
        # Root level Python files that were moved to the package
        "agent.py",
        "enhanced_agent.py",
        "game2048.py",
        "improved_reward.py",
        "reward_function.py",
        "simplified_agent.py",
        "simplified_main.py",
        "simplified_training.py",
        "training.py",
        
        # Batch/shell scripts that were replaced
        "balanced_exploration.bat",
        "balanced_exploration.sh",
        "dynamic_batch_training.bat",
        "dynamic_batch_training.sh",
        "enhanced_main.py",
        
        # Old log files
        "enhanced_training.log",
        "training.log",
        
        # Lock files
        "uv.lock"
    ]
    
    # Directories to consider removing (ask first)
    obsolete_dirs = [
        # Python cache directories
        "__pycache__",
        
        # Old utility directory (moved to package)
        "utils",
        
        # Old directories with potentially useful files
        # (will prompt before deletion)
        "old"
    ]

    # Remove obsolete files
    print("=== Removing obsolete files ===")
    for file in obsolete_files:
        safe_remove(file)
    
    # Ask about directories
    print("\n=== Directories to consider ===")
    for directory in obsolete_dirs:
        if os.path.isdir(directory):
            answer = input(f"Remove directory '{directory}'? (y/n): ").lower()
            if answer == 'y' or answer == 'yes':
                safe_remove(directory)
    
    # Ask about checkpoints
    print("\n=== Checkpoint directories ===")
    checkpoints = glob.glob("*checkpoints")
    if checkpoints:
        for checkpoint_dir in checkpoints:
            if os.path.isdir(checkpoint_dir):
                answer = input(f"Remove checkpoint directory '{checkpoint_dir}'? (y/n): ").lower()
                if answer == 'y' or answer == 'yes':
                    safe_remove(checkpoint_dir)
    
    # Find and clean up bytecode files
    print("\n=== Finding and removing bytecode files ===")
    bytecode_files = find_files(["*.pyc", "*.pyo", "*/__pycache__/*"])
    for file in bytecode_files:
        safe_remove(file)
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    print("This script will remove obsolete and duplicate files from the project.")
    print("It's recommended to have a backup before proceeding.")
    response = input("Do you want to continue? (y/n): ").lower()
    
    if response == 'y' or response == 'yes':
        main()
    else:
        print("Cleanup cancelled.") 