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

def plot_samples(data_loader, model, n, device, model_type, t_in, t_out, dt, do_fft, cmap='coolwarm', autoreg=True, which="Prediction"):
    model.eval()
    data = None
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == n:
                for inputs, outputs in batch:
                    inputs, outputs = inputs.to(device), outputs.to(device)
                    
                    if model_type == "FNO":
                        inputs = inputs.permute(0,3,1,2)
                        outputs = outputs.permute(0,3,1,2)

                    x_coords = inputs[0, 0, :, :].cpu().numpy().flatten()
                    z_coords = inputs[0, 1, :, :].cpu().numpy().flatten()
                    temp_values_t_in = inputs[0, 4, :, :].cpu().numpy().flatten()
                    temp_values_t_out = outputs[0, 0, :, :].cpu().numpy().flatten()

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

                    predictions = predictions.squeeze(1).cpu().numpy().flatten()
                    predictions = predictions * (max_data - min_data) + min_data

                    data = (x_coords, z_coords, temp_values_t_out, predictions)
                    break
                break
    return data

def plot_comparison(x_coords, z_coords, temp_values_t_out, cno_predictions, fno_predictions, nx, nz, do_fft=False):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), dpi=200)

    if do_fft:
        def compute_fft(data):
            data_grid = np.zeros((nx, nz))
            x_indices = np.searchsorted(np.unique(x_coords), x_coords)
            z_indices = np.searchsorted(np.unique(z_coords), z_coords)
            data_grid[x_indices, z_indices] = data
            fft_data = np.fft.fftshift(np.fft.fft2(data_grid))
            return np.abs(fft_data)

        temp_values_t_out = compute_fft(temp_values_t_out)
        cno_predictions = compute_fft(cno_predictions)
        fno_predictions = compute_fft(fno_predictions)

        axs[0].imshow(temp_values_t_out.T, cmap='seismic', origin='lower', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
        axs[0].set_title("FFT of Ground Truth Temperatures", fontsize=20)
        axs[0].set_xlabel('k_x')
        axs[0].set_ylabel('k_z')

        axs[1].imshow(cno_predictions.T, cmap='seismic', origin='lower', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
        axs[1].set_title("FFT of Predicted Temperatures CNO", fontsize=20)
        axs[1].set_xlabel('k_x')

        c3 = axs[2].imshow(fno_predictions.T, cmap='seismic', origin='lower', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
        axs[2].set_title("FFT of Predicted Temperatures FNO", fontsize=20)
        axs[2].set_xlabel('k_x')
    else:
        axs[0].imshow(temp_values_t_out.reshape(nx, nz).T, cmap='coolwarm', origin='lower')
        axs[0].set_title("Ground Truth Temperatures", fontsize=20)
        axs[0].set_ylabel('z [km]', fontsize=14)

        axs[1].imshow(cno_predictions.reshape(nx, nz).T, cmap='coolwarm', origin='lower')
        axs[1].set_title("Predicted Temperatures CNO", fontsize=20)

        c3 = axs[2].imshow(fno_predictions.reshape(nx, nz).T, cmap='coolwarm', origin='lower')
        axs[2].set_title("Predicted Temperatures FNO", fontsize=20)

        for ax in axs:
            m2km = lambda x, _: f'{x/1000:g}'
            ax.xaxis.set_major_formatter(m2km)
            ax.yaxis.set_major_formatter(m2km)
            ax.set_xlabel('x [km]', fontsize=14)

    fig.subplots_adjust(left=0.05, right=0.85, wspace=0.1)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(c3, cax=cbar_ax)

    plt.show()
    plt.savefig('output_comparison.png' if not do_fft else 'output_comparison_fft.png')
    print("Figure saved")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CNO model
# cno_model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_new/model.pkl'
cno_model_path = 'TrainedModels/CNO_straka_bubble_dt_60_new/model.pkl'
cno_model = torch.load(cno_model_path, map_location=torch.device(device))
cno_model.eval()

# Load FNO model
# fno_model_path = 'TrainedModels/FNO_straka_bubble_0_to_900_new/model.pkl'
fno_model_path = 'TrainedModels/FNO_straka_bubble_dt_60_new/model.pkl'
fno_model = torch.load(fno_model_path, map_location=torch.device(device))
fno_model.eval()

autoreg = True
t_in = 0
t_out = 900
dt = 60
do_fft = True
n = 0

# CNO best on 15, worst on 91
# FNO best on 98, worst on 53

if autoreg:
    test_loader_cno = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="CNO", t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
    test_loader_fno = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="FNO", t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader_cno = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type="CNO", dt=dt, normalize=False), batch_size=1, shuffle=False)
    test_loader_fno = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type="FNO", dt=dt, normalize=False), batch_size=1, shuffle=False)

# Get data from CNO model
x_coords, z_coords, temp_values_t_out, cno_predictions = plot_samples(test_loader_cno, cno_model, n, device, "CNO", t_in, t_out, dt, do_fft, cmap='coolwarm', autoreg=autoreg, which="Prediction")

# Get data from FNO model
_, _, _, fno_predictions = plot_samples(test_loader_fno, fno_model, n, device, "FNO", t_in, t_out, dt, do_fft, cmap='coolwarm', autoreg=autoreg, which="Prediction")

# Plot the comparison
x_unique = np.unique(x_coords)
z_unique = np.unique(z_coords)
nx, nz = len(x_unique), len(z_unique)
plot_comparison(x_coords, z_coords, temp_values_t_out, cno_predictions, fno_predictions, nx, nz, do_fft)
