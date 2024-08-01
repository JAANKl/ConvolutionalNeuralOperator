import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
from Dataloaders import resize_and_downsample

def load_and_process_data(file_path, index, timesteps, resolution=256):
    data = {}
    ds = xr.open_mfdataset(file_path + str(index) + "_fields.nc")
    x_values = ds.coords['x'].values
    z_values = ds.coords['z'].values

    for t in timesteps:
        temp_values = torch.tensor(ds['temperature_anomaly'].isel(t=t//60).values, dtype=torch.float32)
        resized_temp_values = resize_and_downsample(temp_values.squeeze(0), (512, 128), (resolution, resolution))
        data[t] = resized_temp_values

    return x_values, z_values, data

def plot_comparison_true_data(file_path, index, timesteps, resolution=256):
    x_coords, z_coords, data_dict = load_and_process_data(file_path, index, timesteps, resolution)

    fig, axs = plt.subplots(1, len(data_dict), figsize=(20, 8), dpi=200)
    x_unique = np.unique(x_coords)
    z_unique = np.unique(z_coords)
    nx, nz = len(x_unique), len(z_unique)

    for i, t in enumerate(timesteps):
        ax = axs[i]
        temp_values = data_dict[t]
        im = ax.imshow(temp_values.T, cmap='coolwarm', origin='lower')
        ax.set_title(f"t={t}", fontsize=20)
        ax.set_xlabel('x [km]')
        if i == 0:
            ax.set_ylabel('z [km]')
        
        m2km = lambda x, _: f'{x/1000:g}'
        ax.xaxis.set_major_formatter(m2km)
        ax.yaxis.set_major_formatter(m2km)

    fig.subplots_adjust(left=0.05, right=0.85, wspace=0.2)
    cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.4])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
    plt.savefig('true_data_comparison.png')
    print("Figure saved")

# Example usage
file_path = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_"
index = 994  # Example index, replace with your actual index
timesteps = [0, 300, 600, 900]  # Example timesteps
resolution = 256  # Desired resolution

plot_comparison_true_data(file_path, index, timesteps, resolution)
