import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def plot_samples(data_loader, model, n, device, model_type, t_in, t_out, dt, do_fft, cmap='coolwarm', autoreg=True):
    model.eval()
    assert t_in < t_out
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == n:
                for inputs, outputs in batch:
                    inputs, outputs = inputs.to(device), outputs.to(device)
                    
                    plot_error = False
                    
                    if model_type == "FNO":
                        inputs = inputs.permute(0,3,1,2)
                        outputs = outputs.permute(0,3,1,2)

                    # Correct data extraction and reshaping based on how your dataloader formats the batch
                    x_coords = inputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming first channel is x-coordinates
                    z_coords = inputs[0, 1, :, :].cpu().numpy().flatten()  # Assuming second channel is y-coordinates
                    temp_values_t_in = inputs[0, 4, :, :].cpu().numpy().flatten()  # Assuming fifth channel is temperature
                    temp_values_t_out = outputs[0, 0, :, :].cpu().numpy().flatten()  # Assuming outputs are correctly squeezed
                    
                    # Normalizing
#                     inputs = (inputs + 28)/28
#                     outputs = (outputs + 28)/28

                    # For new models:
                    inputs[0, 0, :, :] = (inputs[0, 0, :, :] - min_x)/(max_x - min_x)
                    inputs[0, 1, :, :] = (inputs[0, 1, :, :] - min_z)/(max_z - min_z)
                    inputs[0, 2, :, :] = (inputs[0, 2, :, :] - min_viscosity)/(max_viscosity - min_viscosity)
                    inputs[0, 3, :, :] = (inputs[0, 3, :, :] - min_diffusivity)/(max_diffusivity - min_diffusivity)
                    inputs[0, 4, :, :] = (inputs[0, 4, :, :] - min_data)/(max_data - min_data)
                    # outputs = (outputs - min_data)/(max_data - min_data)
                    
                    # For old models:
                    # min_val_in = temp_values_t_in.min()
                    # max_val_in = temp_values_t_in.max()
                    # min_val_out = temp_values_t_out.min()
                    # max_val_out = temp_values_t_out.max()
                    # inputs = (inputs - min_val_in)/(max_val_in - min_val_in)
                    # outputs = (outputs - min_val_out)/(max_val_out - min_val_out)
            
                    
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
                    # predictions = predictions * 28 - 28
                    predictions = predictions * (max_data - min_data) + min_data
                    # For old models:
                    # predictions = predictions * (max_val_out - min_val_out) + min_val_out


                    fig, axes = plt.subplots(1, 3, figsize=(26, 8), sharex=True, sharey=True)
                    
                    
                    if do_fft:
                        x_unique = np.unique(x_coords)
                        z_unique = np.unique(z_coords)

                        nx, nz = len(x_unique), len(z_unique)

                        x_indices = np.searchsorted(x_unique, x_coords)
                        z_indices = np.searchsorted(z_unique, z_coords)

                        temp_in_grid = np.zeros((nx, nz))
                        temp_in_grid[x_indices, z_indices] = temp_values_t_in
                        temp_in_fft = np.fft.fftshift(np.fft.fft2(temp_in_grid))
                        temp_in_fft = np.log(np.abs(temp_in_fft) + 1)
                        
                        temp_out_grid = np.zeros((nx, nz))
                        temp_out_grid[x_indices, z_indices] = temp_values_t_out
                        temp_out_fft = np.fft.fftshift(np.fft.fft2(temp_out_grid))
                        temp_out_fft = np.log(np.abs(temp_out_fft) + 1)
                        
                        predictions_grid = np.zeros((nx, nz))
                        predictions_grid[x_indices, z_indices] = predictions
                        predictions_fft = np.fft.fftshift(np.fft.fft2(predictions_grid))
                        predictions_fft = np.log(np.abs(predictions_fft) + 1)
                        
                        
                        # Input temperature distribution
                        sc1 = axes[0].imshow(temp_in_fft, cmap='hot', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2])
                        axes[0].set_title("FFT of Input Temperatures")
                        axes[0].set_xlabel('k_x')
                        axes[0].set_ylabel('k_z')
                        fig.colorbar(sc1, ax=axes[0])

                        if plot_error:
                            sc2 = axes[1].imshow(temp_out_fft - predictions_fft, cmap='hot', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2])
                            axes[1].set_title("FFT of Error " + model_type)
                        else:
                            # Prediction temperature distribution
                            sc2 = axes[1].imshow(predictions_fft, cmap='hot', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2])
                            axes[1].set_title("FFT of Predicted Temperatures " + model_type)
                        axes[1].set_xlabel('k_x')
                        axes[1].set_ylabel('k_z')
                        fig.colorbar(sc2, ax=axes[1])

                        # Output temperature distribution
                        sc3 = axes[2].imshow(temp_out_fft, cmap='hot', interpolation='nearest', extent=[-nx//2, nx//2, -nz//2, nz//2])
                        axes[2].set_title("FFT of Ground Truth Temperatures")
                        axes[2].set_xlabel('k_x')
                        axes[2].set_ylabel('k_z')
                        fig.colorbar(sc3, ax=axes[2])

                        plt.tight_layout()
                        plt.show()
                        plt.savefig('test_output_plot.png')
                        
                        
                    else:

                        nx, nz = len(np.unique(x_coords)), len(np.unique(z_coords))

                        # Input temperature distribution
                        sc1 = axes[0].imshow(temp_values_t_in.reshape(nx, nz).T, cmap=cmap, interpolation='nearest', origin="lower")
                        axes[0].set_title("Input Temperatures")
                        axes[0].set_xlabel('X coordinate')
                        axes[0].set_ylabel('Z coordinate')
                        fig.colorbar(sc1, ax=axes[0])
                        
                        if plot_error:
                            sc2 = axes[1].imshow(temp_values_t_out.reshape(nx, nz).T - predictions.reshape(nx, nz).T, cmap=cmap, interpolation='nearest', origin="lower")
                            axes[1].set_title("Error " + model_type)
                        else:
                            # Prediction temperature distribution
                            sc2 = axes[1].imshow(predictions.reshape(nx, nz).T, cmap=cmap, interpolation='nearest', origin="lower")
                            axes[1].set_title("Predicted Temperatures " + model_type)
                        axes[1].set_xlabel('X coordinate')
                        axes[1].set_ylabel('Z coordinate')
                        fig.colorbar(sc2, ax=axes[1])

                        # Output temperature distribution
                        sc3 = axes[2].imshow(temp_values_t_out.reshape(nx, nz).T, cmap=cmap, interpolation='nearest', origin="lower")
                        axes[2].set_title("Ground Truth Temperatures")
                        axes[2].set_xlabel('X coordinate')
                        axes[2].set_ylabel('Z coordinate')
                        fig.colorbar(sc3, ax=axes[2])

                        plt.tight_layout()
                        plt.show()
                        plt.savefig('test_output_plot.png')
                        
                        # # deprecated scatterplots:
                        # # Input temperature distribution
                        # sc1 = axes[0].scatter(x_coords, z_coords, c=temp_values_t_in, cmap=cmap, s=13)
                        # axes[0].set_title("Input Temperatures")
                        # axes[0].set_xlabel('X coordinate')
                        # axes[0].set_ylabel('Z coordinate')
                        # fig.colorbar(sc1, ax=axes[0])
                        
                        # if plot_error:
                        #     sc2 = axes[1].scatter(x_coords, z_coords, c=temp_values_t_out - predictions, cmap=cmap, s=13)
                        #     axes[1].set_title("Error " + model_type)
                        # else:
                        #     # Prediction temperature distribution
                        #     sc2 = axes[1].scatter(x_coords, z_coords, c=predictions, cmap=cmap, s=13)
                        #     axes[1].set_title("Predicted Temperatures " + model_type)
                        # axes[1].set_xlabel('X coordinate')
                        # axes[1].set_ylabel('Z coordinate')
                        # fig.colorbar(sc2, ax=axes[1])

                        # # Output temperature distribution
                        # sc3 = axes[2].scatter(x_coords, z_coords, c=temp_values_t_out, cmap=cmap, s=13)
                        # axes[2].set_title("Ground Truth Temperatures")
                        # axes[2].set_xlabel('X coordinate')
                        # axes[2].set_ylabel('Z coordinate')
                        # fig.colorbar(sc3, ax=axes[2])

                        # plt.tight_layout()
                        # plt.show()
                        # plt.savefig('test_output_plot.png')
                    break
                print("Finished plotting batch", n)
                break

                    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_new/model.pkl'
# model_path = 'TrainedModels/FNO_straka_bubble_dt_60_normalized_everywhere/model.pkl'
model_type = model_path.split('TrainedModels/')[1][:3]
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
autoreg = True
t_in=0
t_out=900
dt=900
do_fft = False


if autoreg:
    test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type=model_type, t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type=model_type, dt=dt, normalize=False), batch_size=1, shuffle=False)

plot_samples(test_loader, model, n=2, device=device, model_type=model_type, t_in=t_in, t_out=t_out, dt=dt, do_fft=do_fft, cmap='coolwarm', autoreg=autoreg)
