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

    #             average_amplitude += temp_values_t_out / len(data_loader)
    #             average_predicted_amplitude += predictions / len(data_loader)

    # x_unique = np.unique(x_coords)
    # z_unique = np.unique(z_coords)
    # nx, nz = len(x_unique), len(z_unique)
    
    # x_indices = np.searchsorted(x_unique, x_coords)
    # z_indices = np.searchsorted(z_unique, z_coords)

    # average_amplitude_grid = np.zeros((nx, nz))
    # average_amplitude_grid[x_indices, z_indices] = average_amplitude
    # average_amplitude_fft = np.fft.fftshift(np.fft.fft2(average_amplitude_grid))
    
    # average_predicted_amplitude_grid = np.zeros((nx, nz))
    # average_predicted_amplitude_grid[x_indices, z_indices] = average_predicted_amplitude
    # average_predicted_amplitude_fft = np.fft.fftshift(np.fft.fft2(average_predicted_amplitude_grid))
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

# TODO, absolute value first, then average
# Also for variance and mean calculation: make sure to take either absolute value first or abs **2 instead of just **2
# FT then abs, then average

def compute_energy_spectrum(amplitude_fft, nx, nz):
    # Compute the energy spectrum
    energy_spectrum = np.abs(amplitude_fft)**2
    
    # Compute wavenumbers
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    kz = np.fft.fftshift(np.fft.fftfreq(nz))
    kx, kz = np.meshgrid(kx, kz)
    k = np.sqrt(kx**2 + kz**2)
    
    return energy_spectrum, k


def plot_energy_spectrum(energy_spectrum_true, energy_spectrum_cno, energy_spectrum_fno, k):
    # Flatten the arrays
    k = k.flatten()
    energy_spectrum_true = energy_spectrum_true.flatten()
    energy_spectrum_cno = energy_spectrum_cno.flatten()
    energy_spectrum_fno = energy_spectrum_fno.flatten()
    
    
    # Log bins for wavenumbers
    bins = np.logspace(np.log10(np.min(k[k > 0])), np.log10(np.max(k)), num=50)
    
    # Digitize the wavenumbers
    bin_indices = np.digitize(k, bins)
    
    # Compute the average energy for each bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    energy_true_binned = np.array([energy_spectrum_true[bin_indices == i].mean() if np.any(bin_indices == i) else None for i in range(1, len(bins))])
    energy_cno_binned = np.array([energy_spectrum_cno[bin_indices == i].mean() if np.any(bin_indices == i) else None for i in range(1, len(bins))])
    energy_fno_binned = np.array([energy_spectrum_fno[bin_indices == i].mean() if np.any(bin_indices == i) else None for i in range(1, len(bins))])
    
    
    plt.figure(figsize=(8, 6))
    plt.loglog(bin_centers, energy_cno_binned, label="CNO", color="blue")
    plt.loglog(bin_centers, energy_fno_binned, label="FNO", color="red")
    plt.loglog(bin_centers, energy_true_binned, label="True", color="black")
    plt.xlabel('Wavenumber k')
    plt.ylabel('E(k)')
    plt.title('Energy Spectrum')
    plt.legend()
    plt.show()
    plt.savefig('fft_average_energy_vs_wavenumber.png')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CNO model
model_cno_path = 'TrainedModels/CNO_straka_bubble_0_to_900_new/model.pkl'
model_cno = torch.load(model_cno_path, map_location=torch.device(device))
model_cno.eval()

# Load FNO model
model_fno_path = 'TrainedModels/FNO_straka_bubble_0_to_900_new/model.pkl'
model_fno = torch.load(model_fno_path, map_location=torch.device(device))
model_fno.eval()

autoreg = False
t_in = 0
t_out = 900
dt = 900

if autoreg:
    test_loader_cno = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="CNO", t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
    test_loader_fno = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="FNO", t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader_cno = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type="CNO", dt=dt, normalize=False), batch_size=1, shuffle=False)
    test_loader_fno = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type="FNO", dt=dt, normalize=False), batch_size=1, shuffle=False)

true_fft, predicted_fft_cno, nx, nz = process_model(test_loader_cno, model_cno, device, "CNO", t_in, t_out, dt, autoreg=autoreg)
_, predicted_fft_fno, _, _ = process_model(test_loader_fno, model_fno, device, "FNO", t_in, t_out, dt, autoreg=autoreg)
print("Finished processing models")

energy_spectrum_true, k = compute_energy_spectrum(true_fft, nx, nz)
energy_spectrum_cno, _ = compute_energy_spectrum(predicted_fft_cno, nx, nz)
energy_spectrum_fno, _ = compute_energy_spectrum(predicted_fft_fno, nx, nz)

plot_energy_spectrum(energy_spectrum_true, energy_spectrum_cno, energy_spectrum_fno, k)
print("Finished plotting")


