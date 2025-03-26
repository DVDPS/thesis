@echo off
REM Batch file to evaluate DQN with MCTS after training

REM Activate the virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

REM Default values
set CHECKPOINT=models\dueling_dqn\dueling_dqn_per_best.pt
set GAMES=20
set SIMULATIONS=200

REM Parse command line arguments
if not "%1"=="" set CHECKPOINT=%1
if not "%2"=="" set GAMES=%2
if not "%3"=="" set SIMULATIONS=%3

REM Display settings
echo Evaluating DQN with MCTS enhancement
echo Using checkpoint: %CHECKPOINT% (best performing model)
echo Number of games: %GAMES%
echo Number of simulations: %SIMULATIONS%

REM Run only the enhanced MCTS evaluation
echo Running DQN+MCTS evaluation...
python -m src.thesis.utils.evaluation.enhanced_mcts_evaluation --checkpoint %CHECKPOINT% --games %GAMES% --simulations %SIMULATIONS%

REM Deactivate the virtual environment if it was activated
if exist ".venv\Scripts\deactivate.bat" (
    call ".venv\Scripts\deactivate.bat"
) else if exist "venv\Scripts\deactivate.bat" (
    call "venv\Scripts\deactivate.bat"
)

echo Evaluation complete!
echo Results are saved in the results directory.
pause 