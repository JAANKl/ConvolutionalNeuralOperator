from torch.utils.data import Dataset
import xarray as xr
import torch
import numpy as np
import torch.nn.functional as F


# import h5py
import numpy as np
import torch
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



class MinimalDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __len__(self):
        return 1  # minimal example with a single item

    def __getitem__(self, idx):
        print(f"Trying to open file: {self.file_path}")
        try:
            ds = xr.open_dataset(self.file_path, engine='netcdf4')
            print(f"Successfully opened {self.file_path}")
            data = ds['temperature_anomaly'].isel(t=0).values  # minimal example
            ds.close()
            return torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            print(f"Error opening file {self.file_path}: {e}")
            raise


def resize_and_downsample(u, original_shape=(512, 128), target_shape=(256, 256)):
    """
    Resize and downsample the input tensor to a new shape using bilinear interpolation.
    Args:
    - u (torch.Tensor): The input tensor to be resized. Expected to have shape (H, W).
    - original_shape (tuple): The original shape of the input tensor (H, W).
    - target_shape (tuple): The desired shape (H, W) after resizing.
    Returns:
    - torch.Tensor: The resized tensor with shape (1, H, W) where 1 is the channel dimension.
    """
    # Ensure the input tensor is a float tensor (required for interpolate)
    if not u.is_floating_point():
        u = u.float()
    
    # Reshape the input to add a channel dimension (C=1)
    if u.dim() == 2:
        u = u.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif u.dim() == 3:
        u = u.unsqueeze(1)  # Add a channel dimension

    # Use interpolate to resize
    resized_u = F.interpolate(u, size=target_shape, mode='bilinear', align_corners=False)
    # Remove batch and channel dimensions for output
    resized_u = resized_u.squeeze(0).squeeze(0)
    
    return resized_u


class StrakaBubbleDataset(Dataset):
    def __init__(self, which, training_samples, model_type, dt, normalize=True):
        
        self.dt = dt
        self.max_time = 900
        self.model_type = model_type
        self.normalize = normalize
        self.min_data = -28.0
        self.max_data = 0.0
        self.min_x = 0.0
        self.max_x = 25575.0
        self.min_z = 0.0
        self.max_z = 6400.0
        self.min_viscosity = 0.029050784477663294
        self.max_viscosity = 74.91834790806836
        self.min_diffusivity = 0.043989764422611155
        self.max_diffusivity = 74.8587964361266
        self.resolution = 256
        #The file:
        
        self.file_data = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_"
        
        self.N_max = 1024

        self.n_val  = 128
        self.n_test = 128
        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = self.n_val
            self.start = self.N_max - self.n_val - self.n_test
        elif which == "test":
            self.length = self.n_test
            self.start = self.N_max  - self.n_test
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        ds = xr.open_mfdataset(self.file_data + str(index + self.start) + "_fields.nc")
        data_pairs = []
        for t_in in range(0, self.max_time - self.dt + 1, 60):
            t_out = t_in + self.dt
            if t_out > self.max_time:
                break
            temp_values_t_in = ds['temperature_anomaly'].isel(t=t_in//60).values  # u values at t=t_in
            temp_values_t_out = ds['temperature_anomaly'].isel(t=t_out//60).values  # u values at t=t_out
            x_values = ds.coords['x'].values  # x coordinates
            z_values = ds.coords['z'].values  # z coordinates
            x_grid, z_grid = np.meshgrid(x_values, z_values, indexing='ij')
            viscosity = ds.attrs["VISCOSITY"]
            diffusivity = ds.attrs["DIFFUSIVITY"]
            viscosity = viscosity * np.ones_like(x_grid)
            diffusivity = diffusivity * np.ones_like(x_grid)

            # Convert to PyTorch tensors
            inputs = torch.tensor(np.stack([x_grid, z_grid, viscosity, diffusivity, temp_values_t_in], axis=0), dtype=torch.float32)
            labels = torch.tensor(temp_values_t_out, dtype=torch.float32)


            # Resize and downsample TODO: go to 256x256
            inputs = torch.stack([resize_and_downsample(inputs[i, :, :], (512, 128), (self.resolution, self.resolution)) for i in range(inputs.shape[0])])
            labels = resize_and_downsample(labels.squeeze(0), (512, 128), (self.resolution, self.resolution)) 

            # Normalising
            if self.normalize:
                
                # inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
                # labels = (labels - self.min_data)/(self.max_data - self.min_data)
                
                
                inputs[0, :, :] = (inputs[0, :, :] - self.min_x)/(self.max_x - self.min_x)
                inputs[1, :, :] = (inputs[1, :, :] - self.min_z)/(self.max_z - self.min_z)
                inputs[2, :, :] = (inputs[2, :, :] - self.min_viscosity)/(self.max_viscosity - self.min_viscosity)
                inputs[3, :, :] = (inputs[3, :, :] - self.min_diffusivity)/(self.max_diffusivity - self.min_diffusivity)
                inputs[4, :, :] = (inputs[4, :, :] - self.min_data)/(self.max_data - self.min_data)
                labels = (labels - self.min_data)/(self.max_data - self.min_data)

            if self.model_type == "FNO":
                inputs = inputs.permute(1, 2, 0)
                labels = labels.unsqueeze(-1)
            elif self.model_type == "CNO":
                labels = labels.unsqueeze(0)
            else:
                raise(NotImplementedError("Only FNO and CNO supported."))

            data_pairs.append((inputs, labels))

        return data_pairs