import random

import h5py
import numpy as np
import torch
import xarray as xr
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from FNOModules import FNO2d
from CNOModule import CNO
from training.FourierFeatures import FourierFeatures

from torch.utils.data import Dataset

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


#------------------------------------------------------------------------------

# Some functions needed for loading the Navier-Stokes data

import scipy.fft as fft

def samples_fft(u):
    return fft.fft2(u, norm='forward', workers=-1)


def samples_ifft(u_hat):
    return fft.ifft2(u_hat, norm='forward', workers=-1).real

viscosities = np.zeros(1024)
diffusivities = np.zeros(1024)
amplitudes = np.zeros(1024)
x_r = np.zeros(1024)
z_r = np.zeros(1024)
z_c = np.zeros(1024)

for i in range(1024):
    file_data = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_" + str(i) + "_fields.nc"
    ds = xr.open_mfdataset(file_data) 
    viscosities[i] = ds.attrs["VISCOSITY"]
    diffusivities[i] = ds.attrs["DIFFUSIVITY"]
    amplitudes[i] = ds.attrs["AMPLITUDE"]
    x_r[i] = ds.attrs["X_R"]
    z_r[i] = ds.attrs["Z_R"]
    z_c[i] = ds.attrs["Z_C"]

df = pd.DataFrame({
    'Viscosity': viscosities,
    'Diffusivity': diffusivities,
    'Amplitude': amplitudes,
    'X_R': x_r,
    'Z_R': z_r,
    'Z_C': z_c
})

# Save the DataFrame to a CSV file
csv_file_path = 'straka_data.csv'
df.to_csv(csv_file_path, index=False)


plt.figure(figsize=(10, 6))
plt.scatter(viscosities, diffusivities, alpha=0.5, c='blue')
plt.title('Scatterplot of Viscosities vs Diffusivities')
plt.xlabel('Viscosity')
plt.ylabel('Diffusivity')
plt.grid(True)

# Saving the plot to a file
plt.savefig('viscosities_vs_diffusivities.png')

# Optionally display the plot as well
plt.show()
    
    
    
    
    

