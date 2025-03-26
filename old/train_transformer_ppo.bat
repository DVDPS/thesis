@echo off
setlocal enabledelayedexpansion

:: Script to train the Transformer-based PPO agent for 2048 game

:: Default parameters
set OUTPUT_DIR=transformer_ppo_results
set TIMESTEPS=1000000
set EMBED_DIM=512
set NUM_HEADS=8
set NUM_LAYERS=6
set LEARNING_RATE=0.0001
set CLIP_RATIO=0.2
set SEED=42
set MIXED_PRECISION=true
set DATA_PARALLEL=false
set CHECKPOINT=

:: Create the output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Print training configuration
echo === Transformer PPO Training Configuration ===
echo Output directory: %OUTPUT_DIR%
echo Total timesteps: %TIMESTEPS%
echo Embedding dimension: %EMBED_DIM%
echo Number of attention heads: %NUM_HEADS%
echo Number of transformer layers: %NUM_LAYERS%
echo Learning rate: %LEARNING_RATE%
echo Clip ratio: %CLIP_RATIO%
echo Random seed: %SEED%
echo Mixed precision: %MIXED_PRECISION%
echo Data parallel: %DATA_PARALLEL%
if not "%CHECKPOINT%"=="" (
    echo Resuming from checkpoint: %CHECKPOINT%
)
echo ==========================================

:: Build the command
set CMD=python -m src.thesis.train_transformer_ppo --batch-size 64 ^
    --total-timesteps %TIMESTEPS% ^
    --embed-dim %EMBED_DIM% ^
    --num-heads %NUM_HEADS% ^
    --num-layers %NUM_LAYERS% ^
    --learning-rate %LEARNING_RATE% ^
    --clip-ratio %CLIP_RATIO% ^
    --ent-coef 0.02 ^
    --update-epochs 10 ^
    --seed %SEED% ^
    --output-dir %OUTPUT_DIR%

:: Add optional flags
if "%MIXED_PRECISION%"=="true" (
    set CMD=%CMD% --mixed-precision
)

if "%DATA_PARALLEL%"=="true" (
    set CMD=%CMD% --use-data-parallel
)

if not "%CHECKPOINT%"=="" (
    set CMD=%CMD% --checkpoint %CHECKPOINT%
)

:: Print the command
echo Running: %CMD%
echo Training log will be saved to: %OUTPUT_DIR%\training.log

:: Execute the command
%CMD%

:: Print completion message
echo Training completed. Results saved to %OUTPUT_DIR%
echo To view training curves, run: tensorboard --logdir %OUTPUT_DIR%\tensorboard

:: Recommend next steps
echo === Recommended Next Steps ===
echo 1. Evaluate the model performance using the best model:
echo    python -m src.thesis.evaluate ^
echo        --model transformer_ppo ^
echo        --model-path %OUTPUT_DIR%\best_model.pt ^
echo        --num-games 100
echo.
echo 2. Compare with other agents (if available):
echo    python -m src.thesis.compare_agents ^
echo        --agents transformer_ppo standard_ppo dqn ^
echo        --model-paths %OUTPUT_DIR%\best_model.pt ppo_results\best_model.pt dqn_results\best_model.pt ^
echo        --num-games 50 