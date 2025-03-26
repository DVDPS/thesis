@echo off
REM Batch file to train the DQN agent for 2048

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the DQN training script with optimized parameters
python -m src.thesis.train_dqn ^
    --episodes 10000 ^
    --batch-size 128 ^
    --update-freq 4 ^
    --eval-interval 1000 ^
    --eval-episodes 5 ^
    --log-interval 250 ^
    --output-dir dqn_results %*

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo DQN training complete! 