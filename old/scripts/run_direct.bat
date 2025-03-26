@echo off
REM Direct launch using PYTHONPATH
echo Setting PYTHONPATH to include src directory...
set PYTHONPATH=%PYTHONPATH%;src
echo Running python -m thesis.main %*
python -m thesis.main %*
if %ERRORLEVEL% NEQ 0 (
    echo Error running script!
    pause
) 