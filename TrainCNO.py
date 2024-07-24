import copy
import json
import os
import sys
import csv

import pandas as pd
import xarray as xr
import torch
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# from Problems.CNOBenchmarks import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer
from Dataloaders import StrakaBubble
from Neuralop_Losses import H1Loss

print(sys.argv)

# CNO Strakabubble training parameters:

all_dt = True
t_in=0
t_out=900 
dt=900


if len(sys.argv) == 2:
    
    training_properties = {
        # "learning_rate": 0.001, 
        "learning_rate": 0.001, 
        "weight_decay": 1e-7,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 1000,
        "batch_size": 16,
        "exp": 2.1,                # Do we use L1 or L2 errors? Default: L1. L1:1, L2:2, H1:2.1
        "training_samples": 1024-256  # How many training samples?
    }
    model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        # "N_layers": 3,            # Number of (D) & (U) blocks 
        "N_layers": 5,            # Number of (D) & (U) blocks 
        # "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
        # "N_res": 4,               # Number of (R) blocks in the middle networs.
        "N_res": 2,               # Number of (R) blocks in the middle networks.
        "N_res_neck" : 4,         # Number of (R) blocks in the BN
        
        #Other parameters:
        # "in_size": 64,            # Resolution of the computational grid
        "in_size": 256,            # Resolution of the computational grid TODO: change 256
        "retrain": 4,             # Random seed
        "kernel_size": 3,         # Kernel size.
        "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
        "activation": 'cno_lrelu',# cno_lrelu or cno_lrelu_torch or lrelu or 
        
        #Filter properties:
        "cutoff_den": 2.0001,     # Cutoff parameter.
        "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
        "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
    }
    
    #   "which_example" can be 
    #   straka_bubble       : Straka Bubble

    which_example = sys.argv[1]

    # Save the models here:
    folder = "TrainedModels/"+"CNO_"+which_example+"_0_to_900_no"
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

if which_example == "straka_bubble":
    example = StrakaBubble(model_architecture_, device, batch_size, training_samples, model_type="CNO", all_dt=all_dt, t_in=t_in, t_out=t_out, dt=dt)
else:
    raise ValueError()
    
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
elif p == 2.1:
    loss = H1Loss(d=2, reduce_dims=(0,1))
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.2 * epochs)    # Early stopping parameter
counter = 0

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")

    
# Initialize CSV file and writer for training and validation losses
# train_loss_path = os.path.join(folder, 'train_losses.csv')
val_loss_path = os.path.join(folder, 'val_losses.csv')

# with open(train_loss_path, mode='w', newline='') as file:
#     train_writer = csv.writer(file)
#     train_writer.writerow(['Epoch', 'Batch', 'Train Loss'])

with open(val_loss_path, mode='w', newline='') as file:
    val_writer = csv.writer(file)
    val_writer.writerow(['Epoch', 'Validation Loss', 'Train Relative L2 Error', 'Validation Relative L2 Error'])



for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_loss_avg = 0.0
        for step, batch in enumerate(train_loader):
            for input_batch, output_batch in batch:
                optimizer.zero_grad()
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)

                loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch) # relative L1 loss

                loss_f.backward()
                optimizer.step()
                train_loss_avg = train_loss_avg * step / (step + 1) + loss_f.item() / (step + 1)
                tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_loss_avg})

                # with open(train_loss_path, mode='a', newline='') as file:
                #     train_writer = csv.writer(file)
                #     train_writer.writerow([epoch, step, train_loss_avg])
        # writer.add_scalar("train_loss/train_loss", train_loss_avg, epoch)
        
        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0
            
            for step, batch in enumerate(val_loader):
                for input_batch, output_batch in batch:
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)

                    loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)

            for step, batch in enumerate(train_loader):
                for input_batch, output_batch in batch:
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)

                    loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader)

            with open(val_loss_path, mode='a', newline='') as file:
                val_writer = csv.writer(file)
                val_writer.writerow([epoch, test_relative_l2, train_relative_l2, test_relative_l2])
            
            # writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
            # writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                # writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter+=1

        tepoch.set_postfix({'Train loss': train_loss_avg, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_loss_avg) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        
        scheduler.step()

    if counter>patience:
        print("Early Stopping")
        break
