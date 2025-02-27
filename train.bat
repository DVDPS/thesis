@echo off
REM 2048 Reinforcement Learning Training System
REM Usage examples for different training modes

if "%1"=="" (
    echo Usage: train.bat [mode]
    echo Available modes:
    echo   standard   - Run standard PPO training
    echo   simplified - Run simplified training
    echo   enhanced   - Run enhanced agent training
    echo   balanced   - Run balanced exploration training
    echo   evaluate   - Evaluate a trained model
    exit /b
)

if "%1"=="standard" (
    python -m src.thesis.main --mode standard --epochs 2000 --batch-size 64 --output-dir checkpoints/standard %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="simplified" (
    python -m src.thesis.main --mode simplified --epochs 2000 --batch-size 64 --output-dir checkpoints/simplified %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="enhanced" (
    python -m src.thesis.main --mode enhanced --epochs 2000 --batch-size 96 --output-dir checkpoints/enhanced %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="balanced" (
    python -m src.thesis.main --mode balanced --epochs 2000 --batch-size 96 --dynamic-batch --min-batch-size 16 --output-dir checkpoints/balanced %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="curriculum" (
    python -m src.thesis.main --mode enhanced --epochs 500 --curriculum --curriculum-epochs 500 --checkpoint checkpoints/enhanced/best_model.pt --output-dir checkpoints/curriculum %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="evaluate" (
    python -m src.thesis.main --mode enhanced --evaluate --games 20 --checkpoint %2 %3 %4 %5 %6 %7 %8 %9
) else (
    echo Unknown mode: %1
    echo Run without arguments to see available modes
) 