import copy
import json
import os
import sys
import csv

import pandas as pd
import torch
from tqdm import tqdm

from Dataloaders import StrakaBubble

# FNO Strakabubble training parameters:

all_dt = True
t_in = 0
t_out = 900 
dt = 60

def save_checkpoint(model, optimizer, scheduler, epoch, folder):
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(state, os.path.join(folder, 'checkpoint_FNO.pth'))

def load_checkpoint(model, optimizer, scheduler, folder):
    checkpoint = torch.load(os.path.join(folder, 'checkpoint_FNO.pth'))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    epoch = checkpoint['epoch']
    return epoch

if len(sys.argv) == 2:
    training_properties = {
        "learning_rate": 0.001,
        "weight_decay": 1e-8,
        "scheduler_step": 0.97,
        "scheduler_gamma": 10,
        "epochs": 1000,
        "batch_size": 16,
        "exp": 1,
        "training_samples": 1000-256,
    }
    fno_architecture_ = {
        "width": 64,
        "modes": 20,
        "FourierF": 0,  # Number of Fourier Features in the input channels. Default is 0.
        "n_layers": 5,  # Number of Fourier layers
        "padding": 0,
        "include_grid": 1,
        "retrain": 4,  # Random seed
    }
    
    which_example = sys.argv[1]
    folder = "TrainedModels/" + "FNO_" + which_example + "_dt_60_new"
else:
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    fno_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]

if which_example == "straka_bubble":
    example = StrakaBubble(fno_architecture_, device, batch_size, training_samples, model_type="FNO", all_dt=all_dt, t_in=t_in, t_out=t_out, dt=dt)
else:
    raise ValueError("the variable which_example has to be one between darcy")

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([fno_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

model = example.model
n_params = model.print_size()
train_loader = example.train_loader
test_loader = example.val_loader

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# Initialize CSV file and writer for training and validation losses
val_loss_path = os.path.join(folder, 'val_losses.csv')

with open(val_loss_path, mode='w', newline='') as file:
    val_writer = csv.writer(file)
    val_writer.writerow(['Epoch', 'Validation Loss', 'Train Relative L2 Error', 'Validation Relative L2 Error'])

freq_print = 1
if p == 1:
    loss = torch.nn.SmoothL1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
best_model_testing_error = 300
patience = int(0.25 * epochs)
counter = 0
start_epoch = 0

# Load checkpoint if available
checkpoint_path = os.path.join(folder, 'checkpoint_FNO.pth')
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(model, optimizer, scheduler, folder)
    print(f"Resuming training from epoch {start_epoch}")

for epoch in range(start_epoch, epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        for step, batch in enumerate(train_loader):
            for input_batch, output_batch in batch:
                optimizer.zero_grad()
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)

                loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)

                loss_f.backward()
                optimizer.step()
                train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
                tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0

            for step, batch in enumerate(test_loader):
                for input_batch, output_batch in batch:
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)

                    loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_loader)
            
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

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                counter = 0
            else:
                counter +=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()
        
        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()
    
    save_checkpoint(model, optimizer, scheduler, epoch, folder)
    
    if counter > patience:
        print("Early Stopping")
        break
