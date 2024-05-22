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
    
def default_param(network_properties):
    
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


#------------------------------------------------------------------------------

#NOTE:
#All the training sets should be in the folder: data/

#------------------------------------------------------------------------------

class StrakaBubbleDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 750, s=512, in_dist = True):
        
        self.dt = 60
        self.max_time = 900
        # assert self.t_in in 60 * np.array(range(16))
        # assert self.t_out in 60 * np.array(range(16))
        # assert self.t_in < self.t_out
        self.s = s
        self.in_dist = in_dist
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
            # labels = labels.unsqueeze(0).transpose(1, 2)

            # Normalising
            inputs = (inputs - temp_values_t_in.min())/(temp_values_t_in.max() - temp_values_t_in.min())
            # inputs = (inputs - temp_values_t0.mean())/temp_values_t0.std()
            labels = (labels - temp_values_t_out.min())/(temp_values_t_out.max() - temp_values_t_out.min())
            # labels = (labels - temp_values_t300.mean())/temp_values_t300.std()


            inputs = inputs.permute(1, 2, 0)
            labels = labels.unsqueeze(-1)
            data_pairs.append((inputs, labels))



    #         print("inputs: ", inputs.shape)
    #         print("labels: ", labels.shape)

        return data_pairs
    
class StrakaBubblePlottingDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 750, s=512, t_in=0, t_out=900, in_dist = True):
        
        self.t_in = t_in
        self.t_out = t_out
        self.dt = 60
        self.max_time = 900
        assert self.t_in in 60 * np.array(range(16))
        assert self.t_out in 60 * np.array(range(16))
        assert self.t_in < self.t_out
        self.s = s
        self.in_dist = in_dist
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
        # Extracting u values at t=0 and t=300
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
        # labels = labels.unsqueeze(0).transpose(1, 2)

        # Normalising
        inputs = (inputs - temp_values_t_in.min())/(temp_values_t_in.max() - temp_values_t_in.min())
        # inputs = (inputs - temp_values_t0.mean())/temp_values_t0.std()
        labels = (labels - temp_values_t_out.min())/(temp_values_t_out.max() - temp_values_t_out.min())
        # labels = (labels - temp_values_t300.mean())/temp_values_t300.std()


        inputs = inputs.permute(1, 2, 0)
        labels = labels.unsqueeze(-1)


#         print("inputs: ", inputs.shape)
#         print("labels: ", labels.shape)

        return [(inputs, labels)]
    
    
class StrakaBubble:
    def __init__(self, network_properties, device, batch_size, training_samples, in_dist = True, t_in=0, t_out=600):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            s = self.in_size
        else:
            self.in_size = 64
            s = 64

        network_properties = default_param(network_properties)
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 5, 
                            out_channels = 1, 
                            device=device)        
        
        #----------------------------------------------------------------------
        
        #Change number of workers accoirding to your preference
        num_workers = 0
        self.train_loader = DataLoader(StrakaBubbleDataset("training", training_samples), batch_size=batch_size, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(StrakaBubbleDataset("validation", training_samples), batch_size=batch_size, shuffle=False, num_workers=8)
        self.test_loader = DataLoader(StrakaBubbleDataset("test", training_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers)



#Navier-Stokes data:
#   From 0 to 750 : training samples (750)
#   From 1024 - 128 - 128 to 1024 - 128 : validation samples (128)
#   From 1024 - 128 to 1024 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class ShearLayerDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        self.s = s
        self.in_dist = in_dist
        #The file:
        
        if in_dist:
            if self.s==64:
                self.file_data = "data/NavierStokes_64x64_IN.h5" #In-distribution file 64x64               
            else:
                self.file_data = "data/NavierStokes_128x128_IN.h5"   #In-distribution file 128x128
        else:
            self.file_data = "data/NavierStokes_128x128_OUT.h5"  #Out-of_-distribution file 128x128
        
        self.reader = h5py.File(self.file_data, 'r') 
        self.N_max = 1024

        self.n_val  = 128
        self.n_test = 128
        self.min_data = 1.4307903051376343
        self.max_data = -1.4307903051376343
        self.min_model = 2.0603253841400146
        self.max_model= -2.0383243560791016
        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = self.n_val
            self.start = self.N_max - self.n_val - self.n_test
        elif which == "test":
            self.length = self.n_test
            self.start = self.N_max  - self.n_test
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        if self.s == 64 and self.in_dist:
            inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
            labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        else:
            
            inputs = self.reader['Sample_' + str(index + self.start)]["input"][:].reshape(1,1,self.s, self.s)
            labels = self.reader['Sample_' + str(index + self.start)]["output"][:].reshape(1,1, self.s, self.s)
            
            if self.s<128:
                inputs = downsample(inputs, self.s).reshape(1, self.s, self.s)
                labels = downsample(labels, self.s).reshape(1, self.s, self.s)
            else:
                inputs = inputs.reshape(1, 128, 128)
                labels = labels.reshape(1, 128, 128)
            
            inputs = torch.from_numpy(inputs).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)
            
        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        print("inputs: ", inputs.shape)
        print("labels: ", labels.shape)
        
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)
                
        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid    


class ShearLayer:
    def __init__(self, network_properties, device, batch_size, training_samples, in_size = 64, file = "ddsl_N128/", in_dist = True, padding = 4):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            s = self.in_size
        else:
            self.in_size = 64
            s = 64

        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        
        
        #----------------------------------------------------------------------
        
        #Change number of workers accoirding to your preference
        num_workers = 0
        self.train_loader = DataLoader(ShearLayerDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(ShearLayerDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=8)
        self.test_loader = DataLoader(ShearLayerDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Poisson data:
#   From 0 to 1024 : training samples (1024)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class SinFrequencyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        #The file:
        if in_dist:
            self.file_data = "data/PoissonData_64x64_IN.h5"
        else:
            self.file_data = "data/PoissonData_64x64_OUT.h5"

        #Load normalization constants from the TRAINING set:
        self.reader = h5py.File(self.file_data, 'r')
        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]

        self.s = s #Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024+128
            else:
                self.length = 256
                self.start = 0 
        
        #Load different resolutions
        if s!=64:
            self.file_data = "data/PoissonData_NEW_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed.
        self.reader = h5py.File(self.file_data, 'r')
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class SinFrequency:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):

        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        

        #----------------------------------------------------------------------        

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(SinFrequencyDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(SinFrequencyDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(SinFrequencyDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Wave data:
#   From 0 to 512 : training samples (512)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class WaveEquationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, t = 5, s = 64, in_dist = True):
        
        #Default file:       
        if in_dist:
            self.file_data = "data/WaveData_64x64_IN.h5"
        else:
            self.file_data = "data/WaveData_64x64_OUT.h5"

        self.reader = h5py.File(self.file_data, 'r')
        
        #Load normaliation constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        #What time? DEFAULT : t = 5
        self.t = t
                        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024 + 128
            else:
                self.length = 256
                self.start = 0
        
        self.s = s
        if s!=64:
            self.file_data = "data/WaveData_24modes_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed:
        self.reader = h5py.File(self.file_data, 'r') 
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)
        
        #inputs = inputs + 0.05*torch.randn_like(inputs)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class WaveEquation:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        

        #----------------------------------------------------------------------        

        #Change number of workers accoirding to your preference
        num_workers = 8
        
        self.train_loader = DataLoader(WaveEquationDataset("training", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(WaveEquationDataset("validation", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(WaveEquationDataset("test", self.N_Fourier_F, training_samples, 5, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------

#Allen-Cahn data
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class AllenCahnDataset(Dataset):
    def __init__(self, which="training", nf = 0, training_samples = 1024, s=64, in_dist = True):
        
        #Default file:
        if in_dist:
            self.file_data = "data/AllenCahn_64x64_IN.h5"
        else:            
            self.file_data = "data/AllenCahn_64x64_OUT.h5"
        self.reader = h5py.File(self.file_data, 'r')
        
        #Load normalization constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
                
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 256
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 256 + 128
            else:
                self.length = 128
                self.start = 0 
        
        #Default:
        self.N_Fourier_F = nf
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)
        #print(inputs.shape)    

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class AllenCahn:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        

        #----------------------------------------------------------------------        

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(AllenCahnDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AllenCahnDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(AllenCahnDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

##Smooth Transport data
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class ContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized
        
        #Default file:       
        if in_dist:
            self.file_data = "data/ContTranslation_64x64_IN.h5"
        else:
            self.file_data = "data/ContTranslation_64x64_OUT.h5"
        
        print(self.file_data)
        self.reader = h5py.File(self.file_data, 'r') 

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512 + 256
            else:
                self.length = 256
                self.start = 0

        #Default:
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        #print("I AM HERE BRE")
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)


        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class ContTranslation:
    def __init__(self, network_properties, device, batch_size, training_samples = 512,  s = 64, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        

        #----------------------------------------------------------------------    

        #Change number of workers accoirding to your preference
        num_workers = 0
        
        self.train_loader = DataLoader(ContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(ContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(ContTranslationDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Discontinuous Transport data
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class DiscContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized
        
        if in_dist:
            self.file_data = "data/DiscTranslation_64x64_IN.h5"
        else:
            self.file_data = "data/DiscTranslation_64x64_OUT.h5"
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512+256
            else:
                self.length = 256
                self.start = 0

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)


        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class DiscContTranslation:
    def __init__(self, network_properties, device, batch_size, training_samples = 512, s = 64, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        

        #----------------------------------------------------------------------  

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(DiscContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DiscContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(DiscContTranslationDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Compressible Euler data
#   From 0 to 750 : training samples (750)
#   From 750 to 750 + 128 : validation samples (128)
#   From 750 + 128 to 750 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)
            
class AirfoilDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 128, in_dist = True):
        
        #We DO NOT normalize the data in this case
                
        if in_dist:
            self.file_data = "data/Airfoil_128x128_IN.h5"
        else:
            self.file_data = "data/Airfoil_128x128_OUT.h5"
        
        #in_dist = False
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            
        elif which == "validation":
            self.length = 128
            self.start = 750
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 750 + 128
            else:
                self.length = 128
                self.start = 0

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 128, 128)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 128, 128)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class Airfoil:
    def __init__(self, network_properties, device, batch_size, training_samples = 512, s = 128, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        

        #----------------------------------------------------------------------  

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(AirfoilDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AirfoilDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(AirfoilDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Darcy Flow data
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class DarcyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=256, insample=True):

        if insample:
            self.file_data = "data/Darcy_64x64_IN.h5"
        else:
            self.file_data = "data/Darcy_64x64_IN.h5"
        
        
        self.reader = h5py.File(self.file_data, 'r')

        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
                
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = training_samples
        elif which == "testing":
            if insample:
                self.length = 128
                self.start = training_samples + 128
            else:
                self.length = 128
                self.start = 0

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            grid = grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)

        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            grid = FF(grid)
        return grid

class Darcy:
    def __init__(self, network_properties, device, batch_size, training_samples = 512,  s = 64, in_dist = True):
        
        network_properties = default_param(network_properties)
        self.N_Fourier_F = network_properties["FourierF"]
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------
        
        self.model = FNO2d(fno_architecture = network_properties, 
                            in_channels = 1 + 2 * self.N_Fourier_F, 
                            out_channels = 1, 
                            device=device)        

        #----------------------------------------------------------------------  

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(DarcyDataset("training", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DarcyDataset("validation", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(DarcyDataset("testing", self.N_Fourier_F, training_samples, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)
