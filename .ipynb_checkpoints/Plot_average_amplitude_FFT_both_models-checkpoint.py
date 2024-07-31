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

def plot_comparison(true_fft, predicted_fft_cno, predicted_fft_fno, nx, nz):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), dpi=200)

    c1 = axs[0].imshow(np.abs(true_fft), cmap='seismic', interpolation='nearest', extent=[-nx // 2, nx // 2, -nz // 2, nz // 2], norm=LogNorm())
    axs[0].set_title("Ground Truth", fontsize=20)
    axs[0].set_xlabel(r'$k_x$', fontsize=14)
    axs[0].set_ylabel(r'$k_z$', fontsize=14)
    # fig.colorbar(c1, ax=axs[0])

    c2 = axs[1].imshow(np.abs(predicted_fft_cno), cmap='seismic', interpolation='nearest', extent=[-nx // 2, nx // 2, -nz // 2, nz // 2], norm=LogNorm())
    axs[1].set_title("CNO", fontsize=20)
    axs[1].set_xlabel(r'$k_x$', fontsize=14)
    # axs[1].set_ylabel(r'$k_z$', fontsize=14)
    # fig.colorbar(c2, ax=axs[1])

    c3 = axs[2].imshow(np.abs(predicted_fft_fno), cmap='seismic', interpolation='nearest', extent=[-nx // 2, nx // 2, -nz // 2, nz // 2], norm=LogNorm())
    axs[2].set_title("FNO", fontsize=20)
    axs[2].set_xlabel(r'$k_x$', fontsize=14)
    # axs[2].set_ylabel(r'$k_z$', fontsize=14)
    # fig.colorbar(c3, ax=axs[2])

    fig.subplots_adjust(left=0.05, right=0.85, wspace=0.1)
    
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(c3, cax=cbar_ax)  # Use the last image's color mapping
    
    # plt.tight_layout()
    plt.show()
    plt.savefig('fft_average_amplitude_plot_comparison.png')
    print("Finished plotting FFT of average amplitude")

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

true_fft_cno, predicted_fft_cno, nx_cno, nz_cno = process_model(test_loader_cno, model_cno, device, "CNO", t_in, t_out, dt, autoreg=autoreg)
true_fft_fno, predicted_fft_fno, nx_fno, nz_fno = process_model(test_loader_fno, model_fno, device, "FNO", t_in, t_out, dt, autoreg=autoreg)

# We assume that nx_cno == nx_fno and nz_cno == nz_fno for simplicity
plot_comparison(true_fft_cno, predicted_fft_cno, predicted_fft_fno, nx_cno, nz_cno)
