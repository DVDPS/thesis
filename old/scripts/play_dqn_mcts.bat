@echo off
REM Batch file to play a single game with DQN+MCTS

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the play game script with DQN and MCTS
python -c "from src.thesis.utils.visualization.play_game import main; from src.thesis.agents.dqn_agent import DQNAgent; main('%1', int('%2' or 200), float('%3' or 0.5), True, agent_class=DQNAgent)"

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)

echo Game complete! 