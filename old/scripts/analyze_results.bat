@echo off
REM Batch file to analyze evaluation results

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the analysis script with the provided arguments
python -m src.thesis.utils.evaluation.analyze_results %*

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo Analysis complete! 