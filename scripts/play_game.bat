@echo off
REM Batch file to play a single game of 2048

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the play game script with the provided arguments
python -c "from src.thesis.utils.visualization.play_game import main; main('%1', int('%2' or 0), float('%3' or 0.5), True)"

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo Game complete! 