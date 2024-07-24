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
    assert t_in < t_out
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == n:
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

                    fig, ax = plt.subplots()

                    x_unique = np.unique(x_coords)
                    z_unique = np.unique(z_coords)
                    nx, nz = len(x_unique), len(z_unique)
                    
                    if do_fft:
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
                        
                        if which == "Initial Condition":
                            c = ax.imshow(np.abs(temp_in_fft), cmap='gist_rainbow', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
                            ax.set_title("FFT of Input Temperatures", fontsize=20)
                            
                        elif which == "Prediction":
                            c = ax.imshow(np.abs(predictions_fft), cmap='gist_rainbow', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
                            ax.set_title("FFT of Predicted Temperatures " + model_type, fontsize=20)
                            
                        elif which == "Error":
                            c = ax.imshow(np.abs(temp_out_fft - predictions_fft), cmap='gist_rainbow', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
                            ax.set_title("FFT of Error " + model_type, fontsize=20)
                        
                        elif which == "Ground Truth":
                            c = ax.imshow(np.abs(temp_out_fft), cmap='gist_rainbow', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2], norm=LogNorm())
                            ax.set_title("FFT of Ground Truth Temperatures", fontsize=20)
                        
                        else: 
                            raise NotImplementedError("Typo?")
                        
                        ax.set_xlabel(r'$k_x$', fontsize=14)
                        ax.set_ylabel(r'$k_z$', fontsize=14)
                        fig.colorbar(c, ax=ax)
                        plt.tight_layout()
                        plt.show()
                        plt.savefig('test_output_plot.png')
                        
                        
                    else:
                        if which == "Initial Condition":
                            # scatter = ax.scatter(x_coords, z_coords, c=temp_values_t_in, cmap='gnuplot2', s=13)
                            im = ax.imshow(temp_values_t_in.reshape(nx, nz).T, cmap='gnuplot2', origin='lower')
                            ax.set_title("Input Temperatures", fontsize=20)
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.set_label("Temperature anomalies [K]", fontsize=14)
                        
                        elif which == "Prediction":
                            # scatter = ax.scatter(x_coords, z_coords, c=predictions, cmap='gnuplot2', s=13)
                            im = ax.imshow(predictions.reshape(nx, nz).T, cmap='gnuplot2', origin='lower')
                            ax.set_title("Predicted Temperatures " + model_type, fontsize=20)
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.set_label("Temperature anomalies [K]", fontsize=14)
                        
                        elif which == "Error":
                            # scatter = ax.scatter(x_coords, z_coords, c=temp_values_t_out - predictions, cmap='seismic', s=13, vmin=-2, vmax=2)
                            im = ax.imshow(temp_values_t_out.reshape(nx, nz).T - predictions.reshape(nx, nz).T, cmap='gnuplot2', origin='lower')
                            ax.set_title("Absolute Error " + model_type, fontsize=20)
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.set_label("Absolute Error [K]", fontsize=14)
                        
                        elif which == "Ground Truth":
                            # scatter = ax.scatter(x_coords, z_coords, c=temp_values_t_out, cmap='gnuplot2', s=13)
                            im = ax.imshow(temp_values_t_out.reshape(nx, nz).T, cmap='gnuplot2', origin='lower')
                            ax.set_title("Ground Truth Temperatures", fontsize=20)
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.set_label("Temperature anomalies [K]", fontsize=14)
                            
                        else: 
                            raise NotImplementedError("Typo?")
                        
                        m2km = lambda x, _: f'{x/1000:g}'
                        ax.xaxis.set_major_formatter(m2km)
                        ax.yaxis.set_major_formatter(m2km)
                        ax.set_xlabel('x [km]', fontsize=14)
                        ax.set_ylabel('z [km]', fontsize=14)
                        plt.tight_layout()
                        plt.show()
                        plt.savefig('test_output_plot.png')
                        print("Figure saved")
                    break
                print("Finished plotting batch", n)
                break

                    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_new/model.pkl'
# model_path = 'TrainedModels/FNO_straka_bubble_dt_60_normalized_everywhere/model.pkl'
model_type = model_path.split('TrainedModels/')[1][:3]
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
autoreg = False
t_in=0
t_out=900
dt=900
do_fft = False
# which = "Initial Condition"
# which = "Ground Truth"
which = "Prediction"
# which = "Error"

# FNO dt 60:
# lowest error: n=105
# highest error: n=17


if autoreg:
    test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type=model_type, t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type=model_type, dt=dt, normalize=False), batch_size=1, shuffle=False)

plot_samples(test_loader, model, n=4, device=device, model_type=model_type, t_in=t_in, t_out=t_out, dt=dt, do_fft=do_fft, cmap='coolwarm', autoreg=autoreg, which=which)
