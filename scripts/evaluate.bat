@echo off
REM Batch file to run the 2048 MCTS evaluation

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the evaluation script with the provided arguments
python -m src.thesis.run_evaluation %*

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo Evaluation complete! 