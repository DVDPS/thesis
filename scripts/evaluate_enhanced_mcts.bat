@echo off
REM Batch file to run enhanced MCTS evaluation

REM Activate the virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Default values
set CHECKPOINT=models/dueling_dqn/dueling_dqn_per_best.pt
set GAMES=5
set SIMULATIONS=200
set COMPARE=

REM Parse command line arguments
if not "%1"=="" set GAMES=%1
if not "%2"=="" set SIMULATIONS=%2
if "%3"=="compare" set COMPARE=--compare

REM Display settings
echo Using checkpoint: %CHECKPOINT%
echo Number of games: %GAMES%
echo Number of simulations: %SIMULATIONS%
echo Using device: %DEVICE%

REM Run the enhanced MCTS evaluation script
python -m src.thesis.utils.evaluation.enhanced_mcts_evaluation --checkpoint %CHECKPOINT% --games %GAMES% --simulations %SIMULATIONS% %COMPARE%

REM Deactivate the virtual environment if it was activated
if exist .venv\Scripts\deactivate.bat (
    call .venv\Scripts\deactivate.bat
) else if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo Enhanced MCTS evaluation complete! 