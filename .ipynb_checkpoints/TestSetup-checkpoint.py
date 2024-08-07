# import netCDF4

# file_path = '/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_896_fields.nc'

# try:
#     dataset = netCDF4.Dataset(file_path, 'r')
#     print("Successfully opened the file!")
#     print(dataset)
#     dataset.close()
# except Exception as e:
#     print(f"Error opening the file: {e}")

# import xarray as xr

# file_path = '/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_896_fields.nc'

# try:
#     ds = xr.open_mfdataset(file_path)
#     print("Successfully opened the file with xarray!")
#     print(ds)
# except Exception as e:
#     print(f"Error opening the file with xarray: {e}")


# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# from TestSetupDataloader import StrakaBubbleDataset

# # Constants
# t_in = 0
# t_out = 900
# dt = 900
# batch_size = 1

# import xarray as xr
# import torch
# from torch.utils.data import Dataset


# # Save this as MinimalDataset.py


# # Constants
# file_path = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_896_fields.nc"
# batch_size = 1

# # Initialize the dataset
# # test_dataset = MinimalDataset(file_path=file_path)

# # # Initialize the DataLoader
# # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type="CNO", dt=60, normalize=False), batch_size=1, shuffle=False)

# # Function to print the first batch
# def print_first_batch(data_loader):
#     for i, batch in enumerate(data_loader):
#         print(f"Batch {i}:")
#         print("Data:")
#         print(batch)
#         break

# # Print the first batch
# print_first_batch(test_loader)

# import netCDF4

# file_path = '/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_85_fields.nc'

# try:
#     with netCDF4.Dataset(file_path, 'r') as dataset:
#         print("File opened successfully")
# except OSError as e:
#     print(f"Error opening file: {e}")



# import torch
# from torch.utils.data import DataLoader
# from Dataloaders import StrakaBubble

# model_architecture_ = {
        
#         #Parameters to be chosen with model selection:
#         # "N_layers": 3,            # Number of (D) & (U) blocks 
#         "N_layers": 5,            # Number of (D) & (U) blocks 
#         # "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
#         "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
#         # "N_res": 4,               # Number of (R) blocks in the middle networs.
#         "N_res": 2,               # Number of (R) blocks in the middle networks.
#         "N_res_neck" : 4,         # Number of (R) blocks in the BN
        
#         #Other parameters:
#         # "in_size": 64,            # Resolution of the computational grid
#         "in_size": 256,            # Resolution of the computational grid TODO: change 256
#         "retrain": 4,             # Random seed
#         "kernel_size": 3,         # Kernel size.
#         "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
#         "activation": 'cno_lrelu',# cno_lrelu or cno_lrelu_torch or lrelu or 
        
#         #Filter properties:
#         "cutoff_den": 2.0001,     # Cutoff parameter.
#         "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
#         "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
#         "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
#         "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
#     }

# file_data = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_"
# start = 0
# training_samples = 128
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# data_loader = StrakaBubble(model_architecture_, device, batch_size=32, training_samples=1024-256, model_type="CNO", all_dt=True, t_in=0, t_out=900, dt=900).train_loader

# for i, batch in enumerate(data_loader):
#     print(f"Loaded batch {i}")
#     if i == 10:  # Load a few batches to test
#         break

# ------------------------------

# import copy
# import json
# import os
# import sys
# import csv

# import pandas as pd
# import xarray as xr
# import torch
# # from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm


# # from Problems.CNOBenchmarks import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer
# from Dataloaders import StrakaBubble
# from Neuralop_Losses import H1Loss

# print(sys.argv)

# # CNO Strakabubble training parameters:

# all_dt = True
# t_in=0
# t_out=900 
# dt=900


# if len(sys.argv) == 2:
    
#     training_properties = {
#         # "learning_rate": 0.001, 
#         "learning_rate": 0.001, 
#         "weight_decay": 1e-7,
#         "scheduler_step": 10,
#         "scheduler_gamma": 0.98,
#         "epochs": 1000,
#         "batch_size": 32,
#         "exp": 2.1,                # Do we use L1 or L2 errors? Default: L1. L1:1, L2:2, H1:2.1
#         "training_samples": 1024-256  # How many training samples?
#     }
#     model_architecture_ = {
        
#         #Parameters to be chosen with model selection:
#         # "N_layers": 3,            # Number of (D) & (U) blocks 
#         "N_layers": 5,            # Number of (D) & (U) blocks 
#         # "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
#         "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
#         # "N_res": 4,               # Number of (R) blocks in the middle networs.
#         "N_res": 2,               # Number of (R) blocks in the middle networks.
#         "N_res_neck" : 4,         # Number of (R) blocks in the BN
        
#         #Other parameters:
#         # "in_size": 64,            # Resolution of the computational grid
#         "in_size": 256,            # Resolution of the computational grid TODO: change 256
#         "retrain": 4,             # Random seed
#         "kernel_size": 3,         # Kernel size.
#         "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
#         "activation": 'cno_lrelu',# cno_lrelu or cno_lrelu_torch or lrelu or 
        
#         #Filter properties:
#         "cutoff_den": 2.0001,     # Cutoff parameter.
#         "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
#         "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
#         "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
#         "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
#     }
    
#     #   "which_example" can be 
#     #   straka_bubble       : Straka Bubble

#     which_example = sys.argv[1]

#     # Save the models here:
#     folder = "TrainedModels/"+"CNO_"+which_example+"_0_to_900_final"
        
# else:
    
#     raise NotImplementedError

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

# learning_rate = training_properties["learning_rate"]
# epochs = training_properties["epochs"]
# batch_size = training_properties["batch_size"]
# weight_decay = training_properties["weight_decay"]
# scheduler_step = training_properties["scheduler_step"]
# scheduler_gamma = training_properties["scheduler_gamma"]
# training_samples = training_properties["training_samples"]
# p = training_properties["exp"]

# if not os.path.isdir(folder):
#     print("Generated new folder")
#     os.mkdir(folder)

# df = pd.DataFrame.from_dict([training_properties]).T
# df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
# df = pd.DataFrame.from_dict([model_architecture_]).T
# df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

# if which_example == "straka_bubble":
#     example = StrakaBubble(model_architecture_, device, batch_size, training_samples, model_type="CNO", all_dt=all_dt, t_in=t_in, t_out=t_out, dt=dt)
# else:
#     raise ValueError()
    
# #-----------------------------------Train--------------------------------------
# model = example.model
# n_params = model.print_size()
# train_loader = example.train_loader #TRAIN LOADER
# val_loader = example.val_loader #VALIDATION LOADER

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
# freq_print = 1

# if p == 1:
#     loss = torch.nn.L1Loss()
# elif p == 2:
#     loss = torch.nn.MSELoss()
# elif p == 2.1:
#     loss = H1Loss(d=2, reduce_dims=(0,1))
    
# best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
# patience = int(0.2 * epochs)    # Early stopping parameter
# counter = 0

# if str(device) == 'cpu':
#     print("------------------------------------------")
#     print("YOU ARE RUNNING THE CODE ON A CPU.")
#     print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
#     print("------------------------------------------")
#     print(" ")

    
# # Initialize CSV file and writer
# validation_tracking_path = os.path.join(folder, 'validation_tracking.csv')
# with open(validation_tracking_path, mode='w', newline='') as file:
#     validation_writer = csv.writer(file)
#     # Writing the headers of CSV file
#     validation_writer.writerow(['Epoch', 'Training Loss', 'Training L2 Relative Error', 'Validation Relative L2 Error'])


# data_loader = StrakaBubble(model_architecture_, device, batch_size=32, training_samples=1024-256, model_type="CNO", all_dt=True, t_in=0, t_out=900, dt=900).train_loader

# for i, batch in enumerate(data_loader):
#     print(f"Loaded batch {i}")
#     if i == 10:  # Load a few batches to test
#         break


# for epoch in range(epochs):
#     with tqdm(unit="batch", disable=False) as tepoch:
        
#         model.train()
#         tepoch.set_description(f"Epoch {epoch}")
#         train_loss_avg = 0.0
#         print("HERE_1")
#         for step, batch in enumerate(train_loader):
#             print("HERE_2")
#             for input_batch, output_batch in batch:
#                 print("HERE_3")
#                 optimizer.zero_grad()
#                 input_batch = input_batch.to(device)
#                 output_batch = output_batch.to(device)
#                 output_pred_batch = model(input_batch)

#                 loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch) # relative L1 loss

#                 loss_f.backward()
#                 optimizer.step()
#                 train_loss_avg = train_loss_avg * step / (step + 1) + loss_f.item() / (step + 1)
#                 tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_loss_avg})

#         # writer.add_scalar("train_loss/train_loss", train_loss_avg, epoch)
        
#         with torch.no_grad():
#             model.eval()
#             test_relative_l2 = 0.0
#             train_relative_l2 = 0.0
            
#             for step, batch in enumerate(val_loader):
#                 for input_batch, output_batch in batch:
#                     input_batch = input_batch.to(device)
#                     output_batch = output_batch.to(device)
#                     output_pred_batch = model(input_batch)

#                     if which_example == "airfoil": #Mask the airfoil shape
#                         output_pred_batch[input_batch==1] = 1
#                         output_batch[input_batch==1] = 1

#                     loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
#                     test_relative_l2 += loss_f.item()
#             test_relative_l2 /= len(val_loader)

#             for step, batch in enumerate(train_loader):
#                 for input_batch, output_batch in batch:
#                     input_batch = input_batch.to(device)
#                     output_batch = output_batch.to(device)
#                     output_pred_batch = model(input_batch)

#                     loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
#                     train_relative_l2 += loss_f.item()
#             train_relative_l2 /= len(train_loader)
            
#             # writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
#             # writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

#             if test_relative_l2 < best_model_testing_error:
#                 best_model_testing_error = test_relative_l2
#                 best_model = copy.deepcopy(model)
#                 torch.save(best_model, folder + "/model.pkl")
#                 # writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
#                 counter = 0
#             else:
#                 counter+=1

#         tepoch.set_postfix({'Train loss': train_loss_avg, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
#         tepoch.close()

#         with open(folder + '/errors.txt', 'w') as file:
#             file.write("Training Error: " + str(train_loss_avg) + "\n")
#             file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
#             file.write("Current Epoch: " + str(epoch) + "\n")
#             file.write("Params: " + str(n_params) + "\n")
        
#         with open(validation_tracking_path, mode='a', newline='') as file:
#             validation_writer = csv.writer(file)
#             validation_writer.writerow([epoch, train_loss_avg, train_relative_l2, test_relative_l2])
        
#         scheduler.step()

#     if counter>patience:
#         print("Early Stopping")
#         break


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from matplotlib.colors import LogNorm
# import numpy as np
# import matplotlib.pyplot as plt
# from random import randint
# import pickle

# from Dataloaders import StrakaBubble, StrakaBubbleDataset, StrakaBubblePlottingDataset, normalize_data

# min_data = -28.0
# max_data = 0.0
# min_x = 0.0
# max_x = 25575.0
# min_z = 0.0
# max_z = 6400.0
# min_viscosity = 0.029050784477663294
# max_viscosity = 74.91834790806836
# min_diffusivity = 0.043989764422611155
# max_diffusivity = 74.8587964361266


# def plot_samples(data_loader, model, n, device, model_type, t_in, t_out, dt, do_fft, cmap='coolwarm', autoreg=True, which="Prediction"):
#     model.eval()
#     assert t_in < t_out
#     with torch.no_grad():
#         print(f"Length of DataLoader: {len(data_loader)}")
#         for i, batch in enumerate(data_loader):
#             print(i)
#             print(batch)
#             if i>3:
#                 break


# # for step, batch in enumerate(train_loader):
# #         for input_batch, output_batch in batch:

                    

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_new/model.pkl'
# # model_path = 'TrainedModels/FNO_straka_bubble_dt_60_normalized_everywhere/model.pkl'
# model_type = model_path.split('TrainedModels/')[1][:3]
# model = torch.load(model_path, map_location=torch.device(device))
# model.eval()
# autoreg = True
# t_in=0
# t_out=900
# dt=900
# do_fft = False
# # which = "Initial Condition"
# # which = "Ground Truth"
# # which = "Prediction"
# which = "Error"

# # FNO dt 60:
# # lowest error: n=105
# # highest error: n=17


# if autoreg:
#     test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type=model_type, t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
# else:
#     test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type=model_type, dt=dt, normalize=False), batch_size=1, shuffle=False)

# plot_samples(test_loader, model, n=2, device=device, model_type=model_type, t_in=t_in, t_out=t_out, dt=dt, do_fft=do_fft, cmap='coolwarm', autoreg=autoreg, which=which)
#######################################

# import torch
# from torch.utils.data import DataLoader, Dataset
# from Dataloaders import resize_and_downsample
# import xarray as xr
# import numpy as np

# # Minimal dataset class for testing
# class StrakaBubblePlottingDataset(Dataset):
#     def __init__(self, which, training_samples, model_type, t_in=0, t_out=900):
#         self.model_type = model_type
#         self.t_in = t_in
#         self.t_out = t_out
#         self.max_time = 900
#         self.resolution = 256
#         assert self.t_in in 60 * np.array(range(16))
#         assert self.t_out in 60 * np.array(range(16))
#         assert self.t_in < self.t_out
        
#         #The file:
#         self.file_data = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_"
#         self.N_max = 1024
#         self.n_val  = 128
#         self.n_test = 128
        
#         if which == "training":
#             self.length = training_samples
#             self.start = 0
#         elif which == "validation":
#             self.length = self.n_val
#             self.start = self.N_max - self.n_val - self.n_test
#         elif which == "test":
#             self.length = self.n_test
#             self.start = self.N_max  - self.n_test
        
#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         ds = xr.open_mfdataset(self.file_data + str(index + self.start) + "_fields.nc")
#         temp_values_t_in = ds['temperature_anomaly'].isel(t=self.t_in//60).values
#         temp_values_t_out = ds['temperature_anomaly'].isel(t=self.t_out//60).values
#         x_values = ds.coords['x'].values
#         z_values = ds.coords['z'].values
#         x_grid, z_grid = np.meshgrid(x_values, z_values, indexing='ij')
#         viscosity = ds.attrs["VISCOSITY"]
#         diffusivity = ds.attrs["DIFFUSIVITY"]
#         viscosity = viscosity * np.ones_like(x_grid)
#         diffusivity = diffusivity * np.ones_like(x_grid)

#         # Convert to PyTorch tensors
#         inputs = torch.tensor(np.stack([x_grid, z_grid, viscosity, diffusivity, temp_values_t_in], axis=0), dtype=torch.float32)
#         labels = torch.tensor(temp_values_t_out, dtype=torch.float32)

#         # Resize and downsample
#         inputs = torch.stack([resize_and_downsample(inputs[i, :, :], (512, 128), (self.resolution, self.resolution)) for i in range(inputs.shape[0])])
#         labels = resize_and_downsample(labels.squeeze(0), (512, 128), (self.resolution, self.resolution))

#         if self.model_type == "FNO":
#             inputs = inputs.permute(1, 2, 0)
#             labels = labels.unsqueeze(-1)
#         elif self.model_type == "CNO":
#             labels = labels.unsqueeze(0)
#         else:
#             raise NotImplementedError("Only FNO and CNO supported.")

#         return [(inputs, labels)]

# # Simplified function to just iterate through DataLoader
# def test_dataloader_iteration(data_loader):
#     for i, batch in enumerate(data_loader):
#         print(f"Currently on batch {i}")
#         print(f"Batch {i}: {[b for b in batch]}")
#         if i > 3:  # Limit the number of batches printed for brevity
#             break

# # Initialize DataLoader
# test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="CNO", t_in=0, t_out=900), batch_size=16, shuffle=False)

# # Test DataLoader iteration
# test_dataloader_iteration(test_loader)


# import torch
# from torch.utils.data import Dataset, DataLoader


# class SimpleDataset(Dataset):
#     def __init__(self, size):
#         self.data = torch.arange(size).view(-1, 1).float()

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# # Create a simple dataset and DataLoader
# dataset = SimpleDataset(size=100)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# # Test DataLoader iteration
# for i, batch in enumerate(dataloader):
#     print(f"Batch {i}: {batch}")

# print("Finished iterating over DataLoader.")

# import torch
# from torch.utils.data import DataLoader, Dataset
# # from Dataloaders import resize_and_downsample
# import xarray as xr
# import numpy as np


# # Minimal dataset class for testing
# class StrakaBubblePlottingDataset(Dataset):
#     def __init__(self, which, training_samples, model_type, t_in=0, t_out=900):
#         self.model_type = model_type
#         self.t_in = t_in
#         self.t_out = t_out
#         self.max_time = 900
#         self.resolution = 256
#         assert self.t_in in 60 * np.array(range(16))
#         assert self.t_out in 60 * np.array(range(16))
#         assert self.t_in < self.t_out

#         # The file:
#         self.file_data = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_"
#         self.N_max = 1024
#         self.n_val = 128
#         self.n_test = 128

#         if which == "training":
#             self.length = training_samples
#             self.start = 0
#         elif which == "validation":
#             self.length = self.n_val
#             self.start = self.N_max - self.n_val - self.n_test
#         elif which == "test":
#             self.length = self.n_test
#             self.start = self.N_max - self.n_test

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         inputs = np.zeros(1024)
#         labels = np.zeros(1024)

#         return [(inputs, labels)]


# # Simplified function to just iterate through DataLoader
# def dataloader_iteration(data_loader):
#     for i, batch in enumerate(data_loader):
#         print(f"Currently on batch {i}")
#         print(f"Batch {i}: {[b for b in batch]}")
#         if i > 3:  # Limit the number of batches printed for brevity
#             break


# # Initialize DataLoader
# test_loader = DataLoader(
#     StrakaBubblePlottingDataset(which="test", training_samples=128, model_type="CNO", t_in=0, t_out=900), batch_size=16,
#     shuffle=False)

# # Test DataLoader iteration
# dataloader_iteration(test_loader)


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_data(file_path, t_in_index, t_out_index):
    # Load the dataset
    ds = xr.open_mfdataset(file_path + "1023" + "_fields.nc")
    
    # Extract temperatures at specified time indices
    temp_values_t_in = ds['temperature_anomaly'].isel(t=t_in_index).values
    temp_values_t_out = ds['temperature_anomaly'].isel(t=t_out_index).values
    
    # Extract spatial coordinates
    x_values = ds.coords['x'].values
    z_values = ds.coords['z'].values
    
    # Find the ranges of x and z coordinates
    x_min, x_max = x_values.min(), x_values.max()
    z_min, z_max = z_values.min(), z_values.max()
    
    print(f"Range of x values: {x_min} to {x_max}")
    print(f"Range of z values: {z_min} to {z_max}")

# Parameters
file_path = "/cluster/work/math/camlab-data/data_dana/4-Straka_vdIC_uvDT_timeseries_1024samples/samples/data/sample_"  # Update this path
t_in_index = 0  # Example time index for input
t_out_index = 15  # Example time index for output

load_and_plot_data(file_path, t_in_index, t_out_index)








