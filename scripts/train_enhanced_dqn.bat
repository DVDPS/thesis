@echo off
REM Batch file to train the Enhanced Dueling DQN with Prioritized Experience Replay

REM Activate the virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

REM Default values
set EPISODES=15000
set BATCH_SIZE=8192
set LEARNING_RATE=0.00001
set EPSILON_START=1.0
set EPSILON_END=0.05
set EPSILON_DECAY=0.9995
set GAMMA=0.99
set BUFFER_SIZE=500000
set HIDDEN_DIM=1024
set TARGET_UPDATE=500
set UPDATE_FREQ=4
set SAVE_FREQ=500
set CHECKPOINT_DIR=models\enhanced_dqn
set MODEL_NAME=enhanced_dqn_per


REM Parse command line arguments
if not "%1"=="" set EPISODES=%1
if not "%2"=="" set BATCH_SIZE=%2
if not "%3"=="" set LEARNING_RATE=%3
if not "%4"=="" set RESUME_FROM=%4

REM Create checkpoint directory if it doesn't exist
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

REM Display training settings
echo Starting Enhanced Dueling DQN training with PER and Mixed Precision...
echo Episodes: %EPISODES%
echo Batch Size: %BATCH_SIZE%
echo Learning Rate: %LEARNING_RATE%
echo Epsilon: %EPSILON_START% to %EPSILON_END% (decay: %EPSILON_DECAY%)
echo Buffer Size: %BUFFER_SIZE%
echo Hidden Dim: %HIDDEN_DIM%
echo Target Update: Every %TARGET_UPDATE% steps
echo Saving to: %CHECKPOINT_DIR%\%MODEL_NAME%

REM Run the training script
python -m src.thesis.training.train_dqn ^
    --episodes %EPISODES% ^
    --batch_size %BATCH_SIZE% ^
    --learning_rate %LEARNING_RATE% ^
    --epsilon_start %EPSILON_START% ^
    --epsilon_end %EPSILON_END% ^
    --epsilon_decay %EPSILON_DECAY% ^
    --gamma %GAMMA% ^
    --buffer_size %BUFFER_SIZE% ^
    --hidden_dim %HIDDEN_DIM% ^
    --target_update_freq %TARGET_UPDATE% ^
    --update_freq %UPDATE_FREQ% ^
    --save_freq %SAVE_FREQ% ^
    --checkpoint_dir "%CHECKPOINT_DIR%" ^
    --model_name "%MODEL_NAME%" ^
    --use_per ^
    --dueling

REM Deactivate the virtual environment if it was activated
if exist ".venv\Scripts\deactivate.bat" (
    call ".venv\Scripts\deactivate.bat"
) else if exist "venv\Scripts\deactivate.bat" (
    call "venv\Scripts\deactivate.bat"
)

echo Training complete!
pause 