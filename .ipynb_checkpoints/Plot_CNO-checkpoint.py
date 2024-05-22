import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import h5py
import pickle

# Assuming the model and data loading utilities are available in the specified modules
from Problems.CNOBenchmarks import StrakaBubble, StrakaBubbleDataset, StrakaBubblePlottingDataset
from CNOModule import CNO

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset
# test_dataset = StrakaBubbleDataset(which="test", training_samples=128, t_in=0, t_out=300)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



# Path to the saved model
model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_deep/model.pkl'
# model_path = 'TrainedModels/CNO_poisson_paper/model.pkl'

# Load the entire model directly
model = torch.load(model_path, map_location=torch.device(device))
model.eval()

def plot_samples(data_loader, model, n, device, cmap='coolwarm', vmin=None, vmax=None, autoreg=False):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            for inputs, outputs in batch:
                if i == n:
                    inputs, outputs = inputs.to(device), outputs.to(device)

                    # Correct data extraction and reshaping based on how your dataloader formats the batch
                    x_coords = inputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming first channel is x-coordinates
                    z_coords = inputs[0, 1, :, :].cpu().numpy().flatten()  # Assuming second channel is y-coordinates
                    temp_values_t_in = inputs[0, 4, :, :].cpu().numpy().flatten()  # Assuming fifth channel is temperature
                    temp_values_t_out = outputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming outputs are correctly squeezed
                    
                    # predictions = model(inputs).squeeze(1)  # Ensure predictions match dimensions
                    if autoreg:
                        inputs_running = inputs.clone()
                        for _ in range(15):
                            predictions = model(inputs_running)
                            inputs_running[0, 4, :, :] = predictions[0, 0, :, :]
                    else:
                        predictions = model(inputs)
                    predictions = predictions.squeeze(1)
                    predictions = predictions.cpu().numpy().flatten()

                    fig, axes = plt.subplots(1, 3, figsize=(26, 8), sharex=True, sharey=True)

                    # Input temperature distribution
                    sc1 = axes[0].scatter(x_coords, z_coords, c=temp_values_t_in, cmap=cmap, vmin=vmin, vmax=vmax, s=13)
                    axes[0].set_title("Input Temperatures")
                    axes[0].set_xlabel('X coordinate')
                    axes[0].set_ylabel('Y coordinate')
                    fig.colorbar(sc1, ax=axes[0])

                    # Prediction temperature distribution
                    sc2 = axes[1].scatter(x_coords, z_coords, c=predictions, cmap=cmap, vmin=vmin, vmax=vmax, s=13)
                    axes[1].set_title("Predicted Temperatures")
                    axes[1].set_xlabel('X coordinate')
                    axes[1].set_ylabel('Y coordinate')
                    fig.colorbar(sc2, ax=axes[1])

                    # Output temperature distribution
                    sc3 = axes[2].scatter(x_coords, z_coords, c=temp_values_t_out, cmap=cmap, vmin=vmin, vmax=vmax, s=13)
                    axes[2].set_title("Ground Truth Temperatures")
                    axes[2].set_xlabel('X coordinate')
                    axes[2].set_ylabel('Y coordinate')
                    fig.colorbar(sc3, ax=axes[2])

                    plt.tight_layout()
                    plt.show()
                    plt.savefig('CNO_output_plot.png')
                    break

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test_loader = DataLoader(StrakaBubbleDataset(which="training", training_samples=128), batch_size=1, shuffle=False)
test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, t_in=0, t_out=900), batch_size=1, shuffle=False)
plot_samples(test_loader, model, n=31, device=device, cmap='coolwarm', vmin=None, vmax=None, autoreg=False)
