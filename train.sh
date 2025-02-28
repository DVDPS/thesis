#!/bin/bash
# 2048 Reinforcement Learning Training System
# Usage examples for different training modes

if [ "$1" = "" ]; then
    echo
    echo "2048 Reinforcement Learning Training System"
    echo "========================================="
    echo
    echo "Usage: ./train.sh [mode] [options]"
    echo
    echo "Available modes:"
    echo "  standard   - Run standard PPO training"
    echo "  simplified - Run simplified training"
    echo "  enhanced   - Run enhanced agent training"
    echo "  balanced   - Run balanced exploration training"
    echo "  curriculum - Run curriculum learning"
    echo "  mcts       - Run with MCTS enhancement"
    echo "  evaluate   - Evaluate a trained model"
    echo "  compare    - Compare agent with MCTS version"
    echo "  help       - Show more detailed help"
    echo "  debug      - Run in debug mode to check imports"
    echo "  install    - Install the package in development mode"
    echo
    echo "Example:"
    echo "  ./train.sh enhanced --epochs 1000"
    echo "  ./train.sh evaluate checkpoints/enhanced/best_model.pt"
    echo "  ./train.sh mcts --checkpoint checkpoints/enhanced/best_model.pt --mcts-simulations 100"
    echo "  ./train.sh compare --checkpoint checkpoints/enhanced/best_model.pt"
    exit 0
fi

case "$1" in
    "help")
        python run_2048.py --list-modes
        ;;
    "debug")
        python debug_runner.py
        ;;
    "install")
        python run_2048.py --install
        ;;
    "standard")
        python run_2048.py --mode standard --epochs 2000 --batch-size 64 --output-dir checkpoints/standard "${@:2}"
        ;;
    "simplified")
        python run_2048.py --mode simplified --epochs 2000 --batch-size 64 --output-dir checkpoints/simplified "${@:2}"
        ;;
    "enhanced")
        python run_2048.py --mode enhanced --epochs 2000 --batch-size 96 --output-dir checkpoints/enhanced "${@:2}"
        ;;
    "balanced")
        python run_2048.py --mode balanced --epochs 2000 --batch-size 96 --dynamic-batch --min-batch-size 16 --output-dir checkpoints/balanced "${@:2}"
        ;;
    "curriculum")
        python run_2048.py --mode enhanced --epochs 500 --curriculum --curriculum-epochs 500 --checkpoint checkpoints/enhanced/best_model.pt --output-dir checkpoints/curriculum "${@:2}"
        ;;
    "mcts")
        python run_2048.py --mode mcts --evaluate --games 10 --mcts-simulations 50 --checkpoint checkpoints/enhanced/best_model.pt "${@:2}"
        ;;
    "compare")
        python run_2048.py --mode enhanced --evaluate --games 5 --compare-mcts --mcts-simulations 50 --checkpoint "$2" "${@:3}"
        ;;
    "evaluate")
        python run_2048.py --mode enhanced --evaluate --games 20 --checkpoint "$2" "${@:3}"
        ;;
    *)
        echo "Unknown mode: $1"
        echo "Run ./train.sh without arguments to see available modes"
        ;;
esac 