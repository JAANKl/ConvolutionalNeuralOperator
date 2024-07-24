import copy
import json
import os
import sys
import csv

import pandas as pd
import xarray as xr
import torch
from tqdm import tqdm

from Dataloaders import StrakaBubble
from Neuralop_Losses import H1Loss

def save_checkpoint(model, optimizer, scheduler, epoch, folder):
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(state, os.path.join(folder, 'checkpoint.pth'))

def load_checkpoint(model, optimizer, scheduler, folder):
    checkpoint = torch.load(os.path.join(folder, 'checkpoint.pth'))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    epoch = checkpoint['epoch']
    return epoch

print(sys.argv)

# CNO Strakabubble training parameters:

all_dt = True
t_in = 0
t_out = 900 
dt = 900

if len(sys.argv) == 2:
    training_properties = {
        "learning_rate": 0.001,
        "weight_decay": 1e-7,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 1000,
        "batch_size": 16,
        "exp": 2.1,
        "training_samples": 1024-256
    }
    model_architecture_ = {
        "N_layers": 5,
        "channel_multiplier": 32,
        "N_res": 2,
        "N_res_neck": 4,
        "in_size": 256,
        "retrain": 4,
        "kernel_size": 3,
        "FourierF": 0,
        "activation": 'cno_lrelu',
        "cutoff_den": 2.0001,
        "lrelu_upsampling": 2,
        "half_width_mult": 0.8,
        "filter_size": 6,
        "radial_filter": 0,
    }
    which_example = sys.argv[1]
    folder = "TrainedModels/" + "CNO_" + which_example + "_0_to_900_new"
else:
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
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

model = example.model
n_params = model.print_size()
train_loader = example.train_loader
val_loader = example.val_loader

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
elif p == 2.1:
    loss = H1Loss(d=2, reduce_dims=(0, 1))

best_model_testing_error = 1000
patience = int(0.2 * epochs)
counter = 0
start_epoch = 0

# Load checkpoint if available
checkpoint_path = os.path.join(folder, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(model, optimizer, scheduler, folder)
    print(f"Resuming training from epoch {start_epoch}")

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")

val_loss_path = os.path.join(folder, 'val_losses.csv')

with open(val_loss_path, mode='w', newline='') as file:
    val_writer = csv.writer(file)
    val_writer.writerow(['Epoch', 'Validation Loss', 'Train Relative L2 Error', 'Validation Relative L2 Error'])

for epoch in range(start_epoch, epochs):
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

                loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)

                loss_f.backward()
                optimizer.step()
                train_loss_avg = train_loss_avg * step / (step + 1) + loss_f.item() / (step + 1)
                tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_loss_avg})

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

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                counter = 0
            else:
                counter += 1

        tepoch.set_postfix({'Train loss': train_loss_avg, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_loss_avg) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")

        scheduler.step()

    save_checkpoint(model, optimizer, scheduler, epoch, folder)

    if counter > patience:
        print("Early Stopping")
        break
