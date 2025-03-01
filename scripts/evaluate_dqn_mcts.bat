@echo off
REM Batch file to evaluate the DQN agent with MCTS

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the evaluation script with the provided arguments
python -c "from src.thesis.utils.evaluation.mcts_evaluation import main; main('%1', int('%2' or 5), int('%3' or 200))"

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo Evaluation complete! 