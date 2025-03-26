# 2048 MCTS Evaluation System

This project implements a Monte Carlo Tree Search (MCTS) enhanced agent for playing the 2048 game, along with a comprehensive evaluation system to analyze its performance.

## Project Structure

```
├── scripts/                # Batch files and scripts
│   ├── analyze_results.bat # Script to analyze evaluation results
│   ├── comprehensive_eval.bat # Script to run comprehensive evaluation
│   ├── evaluate.bat        # Script to run evaluation
│   ├── play_game.bat       # Script to play a single game
│   └── ...
├── src/thesis/
│   ├── agents/             # Agent implementations
│   │   ├── enhanced_agent.py # Neural network agent
│   │   └── ...
│   ├── environment/        # Game environment
│   │   ├── game2048.py     # 2048 game implementation
│   │   └── ...
│   ├── utils/              # Utility modules
│   │   ├── cli/            # Command-line interface
│   │   │   └── cli.py      # CLI utilities
│   │   ├── evaluation/     # Evaluation utilities
│   │   │   ├── analyze_results.py # Results analysis
│   │   │   ├── comprehensive_eval.py # Comprehensive evaluation
│   │   │   ├── evaluation.py # Agent evaluation
│   │   │   └── mcts_evaluation.py # MCTS-specific evaluation
│   │   ├── mcts/           # MCTS utilities
│   │   │   ├── mcts.py     # MCTS implementation
│   │   │   └── mcts_agent_wrapper.py # MCTS wrapper for agents
│   │   ├── visualization/  # Visualization utilities
│   │   │   ├── game_analysis.py # Game trajectory analysis
│   │   │   ├── mcts_visualization.py # MCTS search visualization
│   │   │   └── play_game.py # Single game visualization
│   │   └── ...
│   ├── config.py           # Configuration settings
│   ├── main.py             # Main entry point
│   └── run_evaluation.py   # Script to run evaluation
└── ...
```

## Features

- Neural network agent for playing 2048
- MCTS enhancement for improved planning
- Comprehensive evaluation system
- Visualization of game trajectories and MCTS search
- Command-line interface for easy experimentation
- Single game visualization for debugging and demonstration
- Results analysis for comparing different configurations

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```
git clone https://github.com/yourusername/thesis.git
cd thesis
```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
   ```
pip install -r requirements.txt
```

## Usage

### Running Evaluation

On Windows, use the provided batch file:

```
scripts\evaluate.bat --checkpoint checkpoints/best_model.pt --mcts-simulations 200 --games 10
```

Or run directly with Python:

```
python -m src.thesis.run_evaluation --checkpoint checkpoints/best_model.pt --mcts-simulations 200 --games 10
```

### Playing a Single Game

To play and visualize a single game:

```
scripts\play_game.bat checkpoints/best_model.pt 200 0.5
```

Where:
- First argument: Path to the model checkpoint
- Second argument: Number of MCTS simulations (0 for regular agent)
- Third argument: MCTS temperature parameter

This will:
1. Play a single game with the specified agent
2. Display the game progress in the console
3. Generate visualizations in the `game_visualization` directory:
   - Board trajectory showing key frames from the game
   - Action distribution chart
   - Reward progression chart

### Analyzing Results

To analyze and compare results from multiple evaluations:

```
scripts\analyze_results.bat --results-dir evaluation_results --output-dir analysis_results
```

This will:
1. Parse all result files in the specified directory
2. Generate comparison visualizations:
   - Average max tile comparison
   - Average score comparison
   - Best max tile comparison
   - Tile distribution comparison
3. Create a summary table with key metrics

### Running Comprehensive Evaluation

To run a comprehensive evaluation with different MCTS configurations:

```
scripts\comprehensive_eval.bat --checkpoint checkpoints/best_model.pt --games 5 --output-dir comprehensive_results
```

This will:
1. Run evaluations with different MCTS simulation counts (0, 50, 100, 200, 400)
2. Save results for each configuration in separate directories
3. Generate comparison visualizations and summary tables
4. Log the entire process in a comprehensive evaluation log

### Command-line Arguments for Evaluation

- `--checkpoint`: Path to model checkpoint (required)
- `--mcts-simulations`: Number of MCTS simulations (default: 200)
- `--mcts-temperature`: MCTS temperature parameter (default: 0.5)
- `--games`: Number of games to play (default: 10)
- `--render`: Whether to render the games (default: False)
- `--max-steps`: Maximum steps per game (default: 1000)
- `--save-trajectories`: Whether to save game trajectories (default: False)
- `--compare`: Compare regular agent with MCTS-enhanced version (default: False)
- `--output-dir`: Directory to save results (default: "evaluation_results")
- `--hidden-dim`: Hidden dimension size for the neural network (default: 256)

### Examples

1. Evaluate MCTS agent with 200 simulations:
   ```
   scripts\evaluate.bat --checkpoint checkpoints/best_model.pt --mcts-simulations 200 --games 10
   ```

2. Compare regular agent with MCTS-enhanced version:
   ```
   scripts\evaluate.bat --checkpoint checkpoints/best_model.pt --mcts-simulations 200 --games 5 --compare
   ```

3. Save game trajectories for analysis:
   ```
   scripts\evaluate.bat --checkpoint checkpoints/best_model.pt --mcts-simulations 200 --games 3 --save-trajectories
   ```

4. Play a single game with MCTS (200 simulations):
   ```
   scripts\play_game.bat checkpoints/best_model.pt 200 0.5
   ```

5. Analyze results from multiple evaluations:
   ```
   scripts\analyze_results.bat --results-dir evaluation_results
   ```

6. Run comprehensive evaluation with different MCTS configurations:
   ```
   scripts\comprehensive_eval.bat --checkpoint checkpoints/best_model.pt --games 5
   ```

## Results

Evaluation results are saved to the specified output directory (default: "evaluation_results") and include:
- Summary statistics (average max tile, average score, etc.)
- Tile distribution
- Game trajectories (if requested)
- Visualizations of MCTS search (if requested)

Analysis results are saved to the specified output directory (default: "analysis_results") and include:
- Comparison visualizations (average max tile, average score, best max tile, tile distribution)
- Summary table with key metrics

Comprehensive evaluation results are saved to the specified output directory (default: "comprehensive_results") and include:
- Results for each MCTS configuration
- Comparison visualizations and summary tables
- Comprehensive evaluation log

## License

[MIT License](LICENSE)
