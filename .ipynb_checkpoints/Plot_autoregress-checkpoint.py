import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from Dataloaders import StrakaBubble, StrakaBubbleDataset, StrakaBubblePlottingDataset

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

def plot_samples(data_loader, model, n, device, model_type, t_in, t_out, dt, autoreg=True):
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

def plot_comparison(true_data, predictions, nx, nz):
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), dpi=200)

    extent = [0, 25575, 0, 6400]
    aspect_ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])

    for i, t in enumerate([300, 600, 900]):
        # True data plots
        axs[0, i].imshow(true_data[t].reshape(nx, nz).T, cmap='coolwarm', origin='lower', extent=extent, aspect=aspect_ratio)
        axs[0, i].set_title(f"Ground Truth t={t} [s]", fontsize=20)
        axs[0, 0].set_ylabel('z [km]', fontsize=20)
        m2km = lambda x, _: f'{x/1000:g}'
        axs[0, i].xaxis.set_major_formatter(m2km)
        axs[0, i].yaxis.set_major_formatter(m2km)
        axs[0, i].set_xlabel('x [km]', fontsize=20)

        # Prediction plots
        pred_key = f"pred_{(i + 1) * 5}"
        axs[1, i].imshow(predictions[t].reshape(nx, nz).T, cmap='coolwarm', origin='lower', extent=extent, aspect=aspect_ratio)
        axs[1, i].set_title(f"FNO t={t} [s]", fontsize=20)
        axs[1, 0].set_ylabel('z [km]', fontsize=20)
        axs[1, i].xaxis.set_major_formatter(m2km)
        axs[1, i].yaxis.set_major_formatter(m2km)
        axs[1, i].set_xlabel('x [km]', fontsize=20)
    
    fig.subplots_adjust(left=0.05, right=0.85, wspace=0.1, hspace=0.3)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axs[0, 0].images[0], cax=cbar_ax)
    cbar.set_label('Termperature Pertubation [K]', fontsize=20)
    
    plt.show()
    plt.savefig('output_comparison_timesteps.png')
    print("Figure saved")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FNO model
fno_model_path = 'TrainedModels/FNO_straka_bubble_dt_60_new/model.pkl'
fno_model = torch.load(fno_model_path, map_location=torch.device(device))
fno_model.eval()

autoreg = True
t_in = 0
timesteps = [300, 600, 900]
dt = 60
n = 98

test_loader_fno = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="FNO", t_in=t_in, t_out=timesteps[-1]), batch_size=1, shuffle=False)

x_coords, z_coords, _, _ = plot_samples(test_loader_fno, fno_model, n, device, "FNO", t_in, timesteps[-1], dt, autoreg=autoreg)
x_unique = np.unique(x_coords)
z_unique = np.unique(z_coords)
nx, nz = len(x_unique), len(z_unique)

true_data = {}
predictions = {}

for i,t in enumerate(timesteps):
    test_loader_fno = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="FNO", t_in=t_in, t_out=timesteps[i]), batch_size=1, shuffle=False)
    true_data[t] = plot_samples(test_loader_fno, fno_model, n, device, "FNO", t_in, t, dt, autoreg=False)[2]
    pred_steps = (t // 300) * 5
    predictions[t] = plot_samples(test_loader_fno, fno_model, n, device, "FNO", t_in, t, dt, autoreg=True)[3]

plot_comparison(true_data, predictions, nx, nz)
