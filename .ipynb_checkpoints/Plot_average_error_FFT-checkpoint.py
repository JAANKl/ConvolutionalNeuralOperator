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


def plot_samples(data_loader, model, n, device, model_type, t_in, t_out, dt, cmap='coolwarm', autoreg=True):
    model.eval()
    assert t_in < t_out
    average_error_fft = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            for inputs, outputs in batch:
                inputs, outputs = inputs.to(device), outputs.to(device)
                
                if model_type == "FNO":
                    inputs = inputs.permute(0,3,1,2)
                    outputs = outputs.permute(0,3,1,2)

                # Correct data extraction and reshaping based on how your dataloader formats the batch
                x_coords = inputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming first channel is x-coordinates
                z_coords = inputs[0, 1, :, :].cpu().numpy().flatten()  # Assuming second channel is y-coordinates
                temp_values_t_in = inputs[0, 4, :, :].cpu().numpy().flatten()  # Assuming fifth channel is temperature
                temp_values_t_out = outputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming outputs are correctly squeezed

                # Normalizing
                inputs[0, 0, :, :] = (inputs[0, 0, :, :] - min_x)/(max_x - min_x)
                inputs[0, 1, :, :] = (inputs[0, 1, :, :] - min_z)/(max_z - min_z)
                inputs[0, 2, :, :] = (inputs[0, 2, :, :] - min_viscosity)/(max_viscosity - min_viscosity)
                inputs[0, 3, :, :] = (inputs[0, 3, :, :] - min_diffusivity)/(max_diffusivity - min_diffusivity)
                inputs[0, 4, :, :] = (inputs[0, 4, :, :] - min_data)/(max_data - min_data)
                
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
                
                # Scale back the data to original scale
                predictions = predictions * (max_data - min_data) + min_data


                x_unique = np.unique(x_coords)
                z_unique = np.unique(z_coords)
                nx, nz = len(x_unique), len(z_unique)
                
                x_indices = np.searchsorted(x_unique, x_coords)
                z_indices = np.searchsorted(z_unique, z_coords)

                temp_in_grid = np.zeros((nx, nz))
                temp_in_grid[x_indices, z_indices] = temp_values_t_in
                temp_in_fft = np.fft.fftshift(np.fft.fft2(temp_in_grid))
                
                temp_out_grid = np.zeros((nx, nz))
                temp_out_grid[x_indices, z_indices] = temp_values_t_out
                temp_out_fft = np.fft.fftshift(np.fft.fft2(temp_out_grid))
                
                predictions_grid = np.zeros((nx, nz))
                predictions_grid[x_indices, z_indices] = predictions
                predictions_fft = np.fft.fftshift(np.fft.fft2(predictions_grid))

                average_error_fft += np.abs(temp_out_fft - predictions_fft)/len(data_loader)

            fig, ax = plt.subplots()
            c = ax.imshow(average_error_fft, cmap='gist_rainbow', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
            ax.set_title("FFT of Average Error " + model_type, fontsize=20)
            ax.set_xlabel(r'$k_x$', fontsize=14)
            ax.set_ylabel(r'$k_z$', fontsize=14)
            fig.colorbar(c, ax=ax)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'fft_average_error_plot_{model_type}.png')
            print("Finished plotting FFT of average error")
            break

                    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_new/model.pkl'
# model_path = 'TrainedModels/FNO_straka_bubble_0_to_900_normalized_everywhere/model.pkl'
# model_path = 'TrainedModels/FNO_straka_bubble_dt_60_normalized_everywhere/model.pkl'
model_type = model_path.split('TrainedModels/')[1][:3]
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
autoreg = False
t_in=0
t_out=900
dt=900

# FNO dt 60:
# lowest error: n=105
# highest error: n=17


if autoreg:
    test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type=model_type, t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type=model_type, dt=dt, normalize=False), batch_size=1, shuffle=False)

plot_samples(test_loader, model, n=4, device=device, model_type=model_type, t_in=t_in, t_out=t_out, dt=dt, cmap='coolwarm', autoreg=autoreg)
