#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Python and pip if not already installed
echo "Installing Python and pip..."
sudo apt-get install -y python3 python3-pip

# Install git if not already installed
echo "Installing git..."
sudo apt-get install -y git

# Clone the repository if not already cloned
if [ ! -d "thesis" ]; then
    echo "Cloning repository..."
    git clone https://github.com/DVDPS/thesis.git
    cd thesis
else
    echo "Repository already exists, skipping clone..."
    cd thesis
fi

# Create and activate virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy
pip install gymnasium
pip install matplotlib

# Make the main script executable
echo "Making scripts executable..."
chmod +x main.py

echo "Setup complete! To run the game:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the game: python main.py" 