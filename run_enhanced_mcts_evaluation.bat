@echo off
echo Running Enhanced MCTS Evaluation

REM Create necessary directories
mkdir logs 2>nul
mkdir results 2>nul

REM Set Python path
set PYTHONPATH=%PYTHONPATH%;.

REM Default configuration
echo Running default enhanced MCTS configuration...
python src/thesis/evaluation/evaluate_enhanced_mcts.py --num_games 50 --num_simulations 100 --temperature 0.5 --c_puct 2.0 --adaptive --verbose

REM High exploration configuration
echo Running high exploration configuration...
python src/thesis/evaluation/evaluate_enhanced_mcts.py --num_games 50 --num_simulations 100 --temperature 0.7 --c_puct 3.0 --adaptive --verbose

REM Conservative configuration
echo Running conservative configuration...
python src/thesis/evaluation/evaluate_enhanced_mcts.py --num_games 50 --num_simulations 100 --temperature 0.3 --c_puct 1.5 --adaptive --verbose

REM High simulation count
echo Running high simulation count configuration...
python src/thesis/evaluation/evaluate_enhanced_mcts.py --num_games 50 --num_simulations 200 --temperature 0.5 --c_puct 2.0 --adaptive --verbose

REM Parameter sweep (uncomment to run - takes a long time)
REM echo Running parameter sweep...
REM python src/thesis/evaluation/evaluate_enhanced_mcts.py --num_games 10 --parameter_sweep

echo All evaluations complete!
echo Results are saved in the results directory.
pause 