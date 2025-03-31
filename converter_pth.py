import torch

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Open file for writing
with open('model_structure.txt', 'w') as f:
    # Write device information
    f.write(f"Using device: {device}\n")
    if torch.cuda.is_available():
        f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA Version: {torch.version.cuda}\n")

    # Load the model
    checkpoint = torch.load('best_cnn_model.pth', map_location=device)

    # Write model structure and parameters
    f.write("\nModel Structure:\n")
    f.write("-" * 50 + "\n")
    for key, param in checkpoint.items():
        f.write(f"\nKey: {key}\n")
        f.write(f"Shape: {param.shape if hasattr(param, 'shape') else 'N/A'}\n")
        f.write(f"Type: {type(param)}\n")
        f.write(f"Device: {param.device if hasattr(param, 'device') else 'N/A'}\n")
        f.write(f"Values: {param[:5] if hasattr(param, '__getitem__') else param}\n")

print("Model information has been written to 'model_structure.txt'")




