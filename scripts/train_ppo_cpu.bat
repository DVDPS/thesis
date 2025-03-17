@echo off
REM Script to train the PPO agent on CPU only
REM Usage: train_ppo_cpu.bat [output_dir]

REM Activate the virtual environment
call C:\Users\david\.virtualenvs\Group-Project-SNA-GQ9aH4Ii\Scripts\activate.bat

REM Change to the repository root (assuming 'scripts' is a subfolder of the repo)
cd /d "%~dp0\.."

REM Add repository root to PYTHONPATH so that the src package is found
set PYTHONPATH=%cd%

REM Set default output directory
if "%1"=="" (
    set OUTPUT_DIR=ppo_cpu_results
) else (
    set OUTPUT_DIR=%1
)

REM Create the output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Running training on CPU only. This will be slower than GPU training.

REM Run training with CPU settings (smaller model for faster training)
python -m src.thesis.train_ppo ^
    --total-timesteps 500000 ^
    --timesteps-per-update 1024 ^
    --batch-size 64 ^
    --hidden-dim 128 ^
    --learning-rate 3e-4 ^
    --gamma 0.995 ^
    --vf-coef 0.5 ^
    --ent-coef 0.01 ^
    --max-grad-norm 0.5 ^
    --update-epochs 4 ^
    --eval-interval 25 ^
    --eval-episodes 5 ^
    --output-dir %OUTPUT_DIR%

echo Training complete. Results saved to %OUTPUT_DIR% 