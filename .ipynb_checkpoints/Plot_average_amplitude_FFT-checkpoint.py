import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pickle

from Dataloaders import StrakaBubble, StrakaBubbleDataset, StrakaBubblePlottingDataset, normalize_data

min_data = -28.0
max_data = 0.0
min_x = 0.0
max_x = 25575.0
min_z = 0.0
max_z = 6400.0
min_viscosity = 0.029050784477663294
max_viscosity = 74.91834790806836
min_diffusivity = 0.043989764422611155
max_diffusivity = 74.8587964361266

def process_model(data_loader, model, device, model_type, t_in, t_out, dt, autoreg=True):
    model.eval()
    assert t_in < t_out
    sum_amplitude_fft = 0
    sum_predicted_amplitude_fft = 0
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            for inputs, outputs in batch:
                inputs, outputs = inputs.to(device), outputs.to(device)
                
                if model_type == "FNO":
                    inputs = inputs.permute(0, 3, 1, 2)
                    outputs = outputs.permute(0, 3, 1, 2)

                # Correct data extraction and reshaping based on how your dataloader formats the batch
                x_coords = inputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming first channel is x-coordinates
                z_coords = inputs[0, 1, :, :].cpu().numpy().flatten()  # Assuming second channel is y-coordinates
                temp_values_t_in = inputs[0, 4, :, :].cpu().numpy().flatten()  # Assuming fifth channel is temperature
                temp_values_t_out = outputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming outputs are correctly squeezed

                # Normalizing
                inputs[0, 0, :, :] = (inputs[0, 0, :, :] - min_x) / (max_x - min_x)
                inputs[0, 1, :, :] = (inputs[0, 1, :, :] - min_z) / (max_z - min_z)
                inputs[0, 2, :, :] = (inputs[0, 2, :, :] - min_viscosity) / (max_viscosity - min_viscosity)
                inputs[0, 3, :, :] = (inputs[0, 3, :, :] - min_diffusivity) / (max_diffusivity - min_diffusivity)
                inputs[0, 4, :, :] = (inputs[0, 4, :, :] - min_data) / (max_data - min_data)
                
                if autoreg:
                    inputs_running = inputs.clone()
                    for _ in range((t_out - t_in) // dt):
                        if model_type == "FNO":
                            inputs_running = inputs_running.permute(0, 2, 3, 1)

                        predictions = model(inputs_running)

                        if model_type == "FNO":
                            inputs_running = inputs_running.permute(0, 3, 1, 2)
                            predictions = predictions.permute(0, 3, 1, 2)

                        inputs_running[0, 4, :, :] = predictions[0, 0, :, :]
                else:
                    if model_type == "FNO":
                        inputs = inputs.permute(0, 2, 3, 1)

                    predictions = model(inputs)

                    if model_type == "FNO":
                        inputs = inputs.permute(0, 3, 1, 2)
                        predictions = predictions.permute(0, 3, 1, 2)

                predictions = predictions.squeeze(1)
                predictions = predictions.cpu().numpy().flatten()
                
                # Scale back the data to original scale
                predictions = predictions * (max_data - min_data) + min_data

                x_unique = np.unique(x_coords)
                z_unique = np.unique(z_coords)
                nx, nz = len(x_unique), len(z_unique)
                
                x_indices = np.searchsorted(x_unique, x_coords)
                z_indices = np.searchsorted(z_unique, z_coords)

                temp_out_grid = np.zeros((nx, nz))
                temp_out_grid[x_indices, z_indices] = temp_values_t_out
                temp_out_fft = np.fft.fftshift(np.fft.fft2(temp_out_grid))
                
                predictions_grid = np.zeros((nx, nz))
                predictions_grid[x_indices, z_indices] = predictions
                predictions_fft = np.fft.fftshift(np.fft.fft2(predictions_grid))

                sum_amplitude_fft += np.abs(temp_out_fft)
                sum_predicted_amplitude_fft += np.abs(predictions_fft)
                n_samples += 1

    average_amplitude_fft = sum_amplitude_fft / n_samples
    average_predicted_amplitude_fft = sum_predicted_amplitude_fft / n_samples
    return average_amplitude_fft, average_predicted_amplitude_fft, nx, nz

def plot_comparison(true_fft, predicted_fft, nx, nz):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), dpi=200)

    c1 = axs[0].imshow(np.abs(true_fft), cmap='seismic', interpolation='nearest', extent=[-nx // 2, nx // 2, -nz // 2, nz // 2], norm=LogNorm())
    axs[0].set_title("Ground Truth", fontsize=20)
    axs[0].set_xlabel(r'$k_x$', fontsize=14)
    axs[0].set_ylabel(r'$k_z$', fontsize=14)

    c2 = axs[1].imshow(np.abs(predicted_fft), cmap='seismic', interpolation='nearest', extent=[-nx // 2, nx // 2, -nz // 2, nz // 2], norm=LogNorm())
    axs[1].set_title("FNO", fontsize=20)
    axs[1].set_xlabel(r'$k_x$', fontsize=14)

    fig.subplots_adjust(left=0.05, right=0.85, wspace=0.1)
    
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(c2, cax=cbar_ax)  # Use the last image's color mapping
    
    plt.show()
    plt.savefig('fft_average_amplitude_plot.png')
    print("Finished plotting FFT of average amplitude")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FNO model
model_path = 'TrainedModels/FNO_straka_bubble_dt_60_new/model.pkl'
model = torch.load(model_path, map_location=torch.device(device))
model.eval()

autoreg = True
t_in = 0
t_out = 900
dt = 60

if autoreg:
    test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="FNO", t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type="FNO", dt=dt, normalize=False), batch_size=1, shuffle=False)

true_fft, predicted_fft, nx, nz = process_model(test_loader, model, device, "FNO", t_in, t_out, dt, autoreg=autoreg)

plot_comparison(true_fft, predicted_fft, nx, nz)
