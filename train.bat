@echo off
REM 2048 Reinforcement Learning Training System
REM Usage examples for different training modes

if "%1"=="" (
    echo.
    echo 2048 Reinforcement Learning Training System
    echo =========================================
    echo.
    echo Usage: train.bat [mode] [options]
    echo.
    echo Available modes:
    echo   standard   - Run standard PPO training
    echo   simplified - Run simplified training
    echo   enhanced   - Run enhanced agent training
    echo   balanced   - Run balanced exploration training
    echo   curriculum - Run curriculum learning
    echo   evaluate   - Evaluate a trained model
    echo   help       - Show more detailed help
    echo   debug      - Run in debug mode to check imports
    echo   install    - Install the package in development mode
    echo.
    echo Example:
    echo   train.bat enhanced --epochs 1000
    echo   train.bat evaluate checkpoints/enhanced/best_model.pt
    exit /b
)

if "%1"=="help" (
    python run_2048.py --list-modes
    exit /b
)

if "%1"=="debug" (
    python debug_runner.py
    exit /b
)

if "%1"=="install" (
    python run_2048.py --install
    exit /b
)

if "%1"=="standard" (
    python run_2048.py --mode standard --epochs 2000 --batch-size 64 --output-dir checkpoints/standard %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="simplified" (
    python run_2048.py --mode simplified --epochs 2000 --batch-size 64 --output-dir checkpoints/simplified %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="enhanced" (
    python run_2048.py --mode enhanced --epochs 2000 --batch-size 96 --output-dir checkpoints/enhanced %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="balanced" (
    python run_2048.py --mode balanced --epochs 2000 --batch-size 96 --dynamic-batch --min-batch-size 16 --output-dir checkpoints/balanced %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="curriculum" (
    python run_2048.py --mode enhanced --epochs 500 --curriculum --curriculum-epochs 500 --checkpoint checkpoints/enhanced/best_model.pt --output-dir checkpoints/curriculum %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="evaluate" (
    python run_2048.py --mode enhanced --evaluate --games 20 --checkpoint %2 %3 %4 %5 %6 %7 %8 %9
) else (
    echo Unknown mode: %1
    echo Run train.bat without arguments to see available modes
) 