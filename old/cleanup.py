#!/usr/bin/env python
import os
import shutil
import argparse
import sys
import platform

def main():
    parser = argparse.ArgumentParser(description="Clean up redundant files after consolidation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    args = parser.parse_args()
    
    # Files that are now redundant and can be removed
    redundant_files = [
        "visualizations.py",
        "enhanced_exploration.py",
        "curriculum_learning.py",
        "analyze_models.py",
        "README_OPTIMIZATION.md", 
        "visualization.py",  # empty file
        "test_model_loading.py"  # functionality moved to main.py
    ]
    
    # Utility directories to create if they don't exist
    ensure_dirs = [
        "utils",
        "checkpoints", 
        "visualizations"
    ]
    
    # Create necessary directories
    for dir_path in ensure_dirs:
        if not os.path.exists(dir_path):
            if args.dry_run:
                print(f"Would create directory: {dir_path}")
            else:
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
    
    # Delete redundant files
    for file_path in redundant_files:
        if os.path.exists(file_path):
            if args.dry_run:
                print(f"Would delete: {file_path}")
            else:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        else:
            print(f"File not found (already removed): {file_path}")
    
    # Rename the new README
    if os.path.exists("README_CONSOLIDATED.md"):
        if args.dry_run:
            print("Would rename README_CONSOLIDATED.md to README_OPTIMIZATION.md")
        else:
            shutil.copy("README_CONSOLIDATED.md", "README_OPTIMIZATION.md")
            os.remove("README_CONSOLIDATED.md")
            print("Renamed README_CONSOLIDATED.md to README_OPTIMIZATION.md")
    
    print("\nCleanup complete!")
    if args.dry_run:
        print("This was a dry run. No files were actually modified.")
    else:
        print("The repository has been tidied up.")
    
    # Detect OS and provide appropriate command instructions
    is_windows = platform.system() == "Windows"
    if is_windows:
        python_cmd = ".venv\\Scripts\\python main.py"
    else:
        python_cmd = "python main.py"
        
    print("\nThe following command-line interface is now available:")
    print(f"  {python_cmd} --help")
    
    # Additional Windows-specific notes
    if is_windows:
        print("\nWindows-specific notes:")
        print("1. Make sure to use '.venv\\Scripts\\python' to run scripts from your virtual environment")
        print("2. If you encounter import errors, try running from the project root directory")
        print("3. For visualizations, you might need to install additional dependencies:")
        print("   .venv\\Scripts\\pip install matplotlib imageio")

if __name__ == "__main__":
    main() 