@echo off
echo Running MCTS Implementation Comparison

REM Create necessary directories
mkdir logs 2>nul
mkdir results 2>nul
mkdir results\plots 2>nul

REM Set Python path
set PYTHONPATH=%PYTHONPATH%;.

REM Run comparison with default settings
echo Running comparison with default settings...
python src/thesis/evaluation/compare_mcts_implementations.py --num_games 50 --num_simulations 100 --temperature 0.5 --adaptive --verbose

REM Run comparison with higher simulation count
echo Running comparison with higher simulation count...
python src/thesis/evaluation/compare_mcts_implementations.py --num_games 50 --num_simulations 200 --temperature 0.5 --adaptive --verbose

echo All comparisons complete!
echo Results are saved in the results directory.
echo Plots are saved in the results/plots directory.
pause 