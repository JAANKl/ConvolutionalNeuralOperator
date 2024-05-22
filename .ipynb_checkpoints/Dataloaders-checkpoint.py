import random

import h5py
import numpy as np
import torch
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from FNOModules import FNO2d
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


def downsample(u, N):
    N_old = u.shape[-2]
    freqs = fft.fftfreq(N_old, d=1 / N_old)
    sel = np.logical_and(freqs >= -N / 2, freqs <= N / 2 - 1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:, :, sel, :][:, :, :, sel]
    u_down = samples_ifft(u_hat_down)
    return u_down

# For Straka Bubble:

def resize_and_downsample(u, original_shape=(512, 128), target_shape=(128, 128)):
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

#------------------------------------------------------------------------------

#Load default parameters:
    
def default_param_FNO(network_properties):
    
    if "modes" not in network_properties:
        network_properties["modes"] = 16
    
    if "width" not in network_properties:
        network_properties["width"] = 32
    
    if "n_layers" not in network_properties:
        network_properties["n_layers"] = 4

    if "padding" not in network_properties:
        network_properties["padding"] = 0
    
    if "include_grid" not in network_properties:
        network_properties["include_grid"] = 1
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    if "retrain" not in network_properties:
        network_properties["retrain"] = 4
    
    return network_properties

def default_param_CNO(network_properties):
    
    if "channel_multiplier" not in network_properties:
        network_properties["channel_multiplier"] = 32
    
    if "half_width_mult" not in network_properties:
        network_properties["half_width_mult"] = 1
    
    if "lrelu_upsampling" not in network_properties:
        network_properties["lrelu_upsampling"] = 2

    if "filter_size" not in network_properties:
        network_properties["filter_size"] = 6
    
    if "out_size" not in network_properties:
        network_properties["out_size"] = 1
    
    if "radial" not in network_properties:
        network_properties["radial_filter"] = 0
    
    if "cutoff_den" not in network_properties:
        network_properties["cutoff_den"] = 2.0001
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    if "retrain" not in network_properties:
        network_properties["retrain"] = 4
    
    if "kernel_size" not in network_properties:
        network_properties["kernel_size"] = 3
    
    if "activation" not in network_properties:
        network_properties["activation"] = 'cno_lrelu'
    
    return network_properties


class StrakaBubbleDataset(Dataset):
    def __init__(self, which, training_samples, model_type, dt):
        
        self.dt = dt
        self.max_time = 900
        self.model_type = model_type
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


            # Resize and downsample
            inputs = torch.stack([resize_and_downsample(inputs[i, :, :], (512, 128), (128, 128)) for i in range(inputs.shape[0])])
            labels = resize_and_downsample(labels.squeeze(0), (512, 128), (128, 128))

            # Normalising
            inputs = (inputs - temp_values_t_in.min())/(temp_values_t_in.max() - temp_values_t_in.min())
            # inputs = (inputs - temp_values_t0.mean())/temp_values_t0.std()
            labels = (labels - temp_values_t_out.min())/(temp_values_t_out.max() - temp_values_t_out.min())
            # labels = (labels - temp_values_t300.mean())/temp_values_t300.std()

            if self.model_type == "FNO":
                inputs = inputs.permute(1, 2, 0)
                labels = labels.unsqueeze(-1)
            elif self.model_type == "CNO":
                labels = labels.unsqueeze(0)
            else:
                raise(NotImplementedError("Only FNO and CNO supported."))
            data_pairs.append((inputs, labels))

        return data_pairs
    
class StrakaBubblePlottingDataset(Dataset):
    def __init__(self, which, training_samples, model_type, t_in=0, t_out=900):
        self.model_type = model_type
        self.t_in = t_in
        self.t_out = t_out
        self.max_time = 900
        assert self.t_in in 60 * np.array(range(16))
        assert self.t_out in 60 * np.array(range(16))
        assert self.t_in < self.t_out
        
        #The file:
        
        self.file_data = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_"
        
        # self.reader = xr.open_mfdataset() 
        
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
        temp_values_t_in = ds['temperature_anomaly'].isel(t=self.t_in//60).values  # u values at t=t_in
        temp_values_t_out = ds['temperature_anomaly'].isel(t=self.t_out//60).values  # u values at t=t_out
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


        # Resize and downsample
        inputs = torch.stack([resize_and_downsample(inputs[i, :, :], (512, 128), (128, 128)) for i in range(inputs.shape[0])])
        labels = resize_and_downsample(labels.squeeze(0), (512, 128), (128, 128))

        # Normalising
        inputs = (inputs - temp_values_t_in.min())/(temp_values_t_in.max() - temp_values_t_in.min())
        # inputs = (inputs - temp_values_t0.mean())/temp_values_t0.std()
        labels = (labels - temp_values_t_out.min())/(temp_values_t_out.max() - temp_values_t_out.min())
        # labels = (labels - temp_values_t300.mean())/temp_values_t300.std()

        if self.model_type == "FNO":
            inputs = inputs.permute(1, 2, 0)
            labels = labels.unsqueeze(-1)
        elif self.model_type == "CNO":
            labels = labels.unsqueeze(0)
        else:
            raise(NotImplementedError("Only FNO and CNO supported."))


        return [(inputs, labels)]
    
    
class StrakaBubble:
    def __init__(self, network_properties, device, batch_size, training_samples, in_dist = True, model_type="FNO", all_dt=True, t_in=0, t_out=900, dt=60):
        
        if model_type == "FNO":
        
            if "in_size" in network_properties:
                self.in_size = network_properties["in_size"]
                s = self.in_size
            else:
                self.in_size = 64
                s = 64

            network_properties = default_param_FNO(network_properties)

            retrain = network_properties["retrain"]
            torch.manual_seed(retrain)

            #----------------------------------------------------------------------

            self.model = FNO2d(fno_architecture = network_properties, 
                                in_channels = 5, 
                                out_channels = 1, 
                                device=device)  
            
        elif model_type == "CNO":
            #Must have parameters: ------------------------------------------------        

            if "in_size" in network_properties:
                self.in_size = network_properties["in_size"]
                assert self.in_size<=512        
            else:
                raise ValueError("You must specify the computational grid size.")

            if "N_layers" in network_properties:
                N_layers = network_properties["N_layers"]
            else:
                raise ValueError("You must specify the number of (D) + (U) blocks.")

            if "N_res" in network_properties:
                    N_res = network_properties["N_res"]        
            else:
                raise ValueError("You must specify the number of (R) blocks.")

            if "N_res_neck" in network_properties:
                    N_res_neck = network_properties["N_res_neck"]        
            else:
                raise ValueError("You must specify the number of (R)-neck blocks.")

            #Load default parameters if they are not in network_properties
            network_properties = default_param_CNO(network_properties)

            #----------------------------------------------------------------------
            kernel_size = network_properties["kernel_size"]
            channel_multiplier = network_properties["channel_multiplier"]
            retrain = network_properties["retrain"]

            #Filter properties: ---------------------------------------------------
            cutoff_den = network_properties["cutoff_den"]
            filter_size = network_properties["filter_size"]
            half_width_mult = network_properties["half_width_mult"]
            lrelu_upsampling = network_properties["lrelu_upsampling"]
            activation = network_properties["activation"]
            ##----------------------------------------------------------------------

            torch.manual_seed(retrain)

            self.model = CNO(in_dim  = 5,     # Number of input channels.
                            in_size = self.in_size,                # Input spatial size
                            N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                            N_res = N_res,                         # Number of (R) Blocks per level
                            N_res_neck = N_res_neck,
                            channel_multiplier = channel_multiplier,
                            conv_kernel=kernel_size,
                            cutoff_den = cutoff_den,
                            filter_size=filter_size,  
                            lrelu_upsampling = lrelu_upsampling,
                            half_width_mult  = half_width_mult,
                            activation = activation).to(device)

            #----------------------------------------------------------------------
        else:
            raise(NotImplementedError("Only FNO and CNO supported."))
            
        #----------------------------------------------------------------------

        #Change number of workers according to your preference
        num_workers = 0
        
        if all_dt:
            self.train_loader = DataLoader(StrakaBubbleDataset("training", training_samples, model_type, dt), batch_size=batch_size, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(StrakaBubbleDataset("validation", training_samples, model_type, dt), batch_size=batch_size, shuffle=False, num_workers=8)
            self.test_loader = DataLoader(StrakaBubbleDataset("test", training_samples, model_type, dt), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            self.train_loader = DataLoader(StrakaBubblePlottingDataset("training", training_samples, model_type, t_in, t_out), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_loader = DataLoader(StrakaBubblePlottingDataset("validation", training_samples, model_type, t_in, t_out), batch_size=batch_size, shuffle=False, num_workers=num_workers)
            self.test_loader = DataLoader(StrakaBubblePlottingDataset("test", training_samples, model_type, t_in, t_out), batch_size=batch_size, shuffle=False, num_workers=num_workers)