import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def list_files_in_directory(directory):
    """List all files in the given directory."""
    files = os.listdir(directory)
    for file in files:
        print(file)
    return files

def plot_data(file_path, times, file_index=0):
    # Construct the full file path
    file_name = f"{file_path}{file_index}_fields.nc"
    print(f"Loading file: {file_name}")
    
    # Load the dataset
    ds = xr.open_dataset(file_name)

    print("-------------------------------File opened successfully--------------------------------")

    print(ds)
    
    # Extract spatial coordinates
    x_values = ds.coords['x'].values
    z_values = ds.coords['z'].values
    
    # Create meshgrid for coordinates
    x_grid, z_grid = np.meshgrid(x_values, z_values, indexing='ij')
    
    # fig, axes = plt.subplots(1, len(times), figsize=(20, 5), sharey=True)
    fig, axes = plt.subplots(1, len(times), figsize=(20, 8), dpi=200)

    extent = [0, 25575, 0, 6400]
    aspect_ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])
    
    for i, t in enumerate(times):
        # Extract temperature anomaly at the specified time
        temp_values = ds['temperature_anomaly'].isel(t=t // 60).values
        
        # Plotting
        ax = axes[i]
        sc = ax.imshow(temp_values.T, cmap='coolwarm', origin='lower', extent=extent, aspect=aspect_ratio)
        ax.set_title(f't = {t} [s]', fontsize=20)
        ax.set_xlabel('x [km]', fontsize=20)
        if i == 0:
            ax.set_ylabel('z [km]', fontsize=20)

        m2km = lambda x, _: f'{x/1000:g}'
        ax.xaxis.set_major_formatter(m2km)
        ax.yaxis.set_major_formatter(m2km)
        ax.set_xlim(0, 25575)
        ax.set_ylim(0, 6400)
    
    fig.subplots_adjust(left=0.05, right=0.85, wspace=0.1)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Temperature Perturbation [K]', fontsize=20)
    
    plt.show()
    plt.savefig("output_nature_run.png")
    print("Figure saved")

# List files in the directory
# directory = "/cluster/work/math/camlab-data/data_dana/0-PASC_plots/nature_run/data/"
directory = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/"
# files = list_files_in_directory(directory)

# Parameters
file_path = directory + "sample_"
times = [0, 300, 600, 900]  # Times in seconds
n = 949

plot_data(file_path, times, file_index=n)
