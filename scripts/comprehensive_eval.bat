@echo off
REM Batch file to run comprehensive evaluation

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the comprehensive evaluation script with the provided arguments
python -m src.thesis.utils.evaluation.comprehensive_eval %*

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo Comprehensive evaluation complete! 