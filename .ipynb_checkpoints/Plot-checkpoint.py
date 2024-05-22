import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import h5py
import pickle

from Dataloaders import StrakaBubble, StrakaBubbleDataset, StrakaBubblePlottingDataset


def plot_samples(data_loader, model, n, device, model_type, t_in, t_out, dt, cmap='coolwarm', autoreg=True):
    model.eval()
    assert t_in < t_out
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            for inputs, outputs in batch:
                if i == n:
                    inputs, outputs = inputs.to(device), outputs.to(device)
                    
                    if model_type == "FNO":
                        inputs = inputs.permute(0,3,1,2)
                        outputs = outputs.permute(0,3,1,2)

                    # Correct data extraction and reshaping based on how your dataloader formats the batch
                    x_coords = inputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming first channel is x-coordinates
                    z_coords = inputs[0, 1, :, :].cpu().numpy().flatten()  # Assuming second channel is y-coordinates
                    temp_values_t_in = inputs[0, 4, :, :].cpu().numpy().flatten()  # Assuming fifth channel is temperature
                    temp_values_t_out = outputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming outputs are correctly squeezed
                    
                    if autoreg:
                        inputs_running = inputs.clone()
                        for _ in range((t_out - t_in) // dt):
                            if model_type == "FNO":
                                inputs_running = inputs_running.permute(0,2,3,1)
                                
                            predictions = model(inputs_running)
                            
                            if model_type == "FNO":
                                inputs_running = inputs_running.permute(0,3,1,2)
                                predictions = predictions.permute(0,3,1,2)
                                
                            inputs_running[0, 4, :, :] = predictions[0, 0, :, :]
                    else:
                        if model_type == "FNO":
                            inputs = inputs.permute(0,2,3,1)
                            
                        predictions = model(inputs)
                        
                        if model_type == "FNO":
                            inputs = inputs.permute(0,3,1,2)
                            predictions = predictions.permute(0,3,1,2)
                            
                            
                    predictions = predictions.squeeze(1)
                    predictions = predictions.cpu().numpy().flatten()

                    fig, axes = plt.subplots(1, 3, figsize=(26, 8), sharex=True, sharey=True)

                    # Input temperature distribution
                    sc1 = axes[0].scatter(x_coords, z_coords, c=temp_values_t_in, cmap=cmap, s=13)
                    axes[0].set_title("Input Temperatures")
                    axes[0].set_xlabel('X coordinate')
                    axes[0].set_ylabel('Y coordinate')
                    fig.colorbar(sc1, ax=axes[0])

                    # Prediction temperature distribution
                    sc2 = axes[1].scatter(x_coords, z_coords, c=predictions, cmap=cmap, s=13)
                    axes[1].set_title("Predicted Temperatures")
                    axes[1].set_xlabel('X coordinate')
                    axes[1].set_ylabel('Y coordinate')
                    fig.colorbar(sc2, ax=axes[1])

                    # Output temperature distribution
                    sc3 = axes[2].scatter(x_coords, z_coords, c=temp_values_t_out, cmap=cmap, s=13)
                    axes[2].set_title("Ground Truth Temperatures")
                    axes[2].set_xlabel('X coordinate')
                    axes[2].set_ylabel('Y coordinate')
                    fig.colorbar(sc3, ax=axes[2])

                    plt.tight_layout()
                    plt.show()
                    plt.savefig('test_output_plot.png')
                    break

                    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = 'TrainedModels/FNO_straka_bubble_dt_60/model.pkl'
model_path = 'TrainedModels/CNO_straka_bubble_0_to_600/model.pkl'
model_type = model_path.split('TrainedModels/')[1][:3]
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
autoreg = True
t_in=0
t_out=600
dt=600

if autoreg:
    test_loader = DataLoader(StrakaBubblePlottingDataset(which="training", training_samples=128, model_type=model_type, t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type=model_type, dt=dt), batch_size=1, shuffle=False)

plot_samples(test_loader, model, n=2, device=device, model_type=model_type, t_in=t_in, t_out=t_out, dt=dt, cmap='coolwarm', autoreg=autoreg)
