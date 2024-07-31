import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Assuming StrakaBubbleDataset and necessary imports are correctly configured
from Dataloaders import StrakaBubbleDataset, StrakaBubblePlottingDataset

min_data = -28.0
max_data = 0.0
min_x = 0.0
max_x = 25575.0
min_z = 0.0
max_z = 6400.0
min_viscosity = 0.029050784477663294
max_viscosity = 74.91834790806836
min_diffusivity = 0.043989764422611155
max_diffusivity = 74.8587964361266

def calculate_l2_error(output, prediction):
    return torch.norm(output - prediction, p=2) / torch.norm(output, p=2)

def plot_error_by_attributes(data_loader, model, model_type, t_in, t_out, dt, autoreg, csv_filename, device):
    viscosity_values = []
    diffusivity_values = []
    errors = []
    
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Viscosity', 'Diffusivity', 'Error'])
    
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                for inputs, outputs in batch:
                    inputs, outputs = inputs.to(device), outputs.to(device)

                    if model_type == "FNO":
                        inputs = inputs.permute(0,3,1,2)
                        outputs = outputs.permute(0,3,1,2)        
                    
                    inputs[0, 0, :, :] = (inputs[0, 0, :, :] - min_x)/(max_x - min_x)
                    inputs[0, 1, :, :] = (inputs[0, 1, :, :] - min_z)/(max_z - min_z)
                    inputs[0, 2, :, :] = (inputs[0, 2, :, :] - min_viscosity)/(max_viscosity - min_viscosity)
                    inputs[0, 3, :, :] = (inputs[0, 3, :, :] - min_diffusivity)/(max_diffusivity - min_diffusivity)
                    inputs[0, 4, :, :] = (inputs[0, 4, :, :] - min_data)/(max_data - min_data)
            
                    
                    if autoreg:
                        inputs_running = inputs.clone()
                        for _ in range((t_out - t_in) // dt):
                            if model_type == "FNO":
                                inputs_running = inputs_running.permute(0,2,3,1)

                            predictions = model(inputs_running)

                            if model_type == "FNO":
                                inputs_running = inputs_running.permute(0,3,1,2)
                                predictions = predictions.permute(0,3,1,2)

                            inputs_running[0, 4, :, :] = predictions[0, 0, :, :]
                    else:
                        if model_type == "FNO":
                            inputs = inputs.permute(0,2,3,1)

                        predictions = model(inputs)

                        if model_type == "FNO":
                            inputs = inputs.permute(0,3,1,2)
                            predictions = predictions.permute(0,3,1,2)


                    predictions = predictions.squeeze(1)
                    
                    # Scale back the data to original scale
                    predictions = predictions * (max_data - min_data) + min_data

                    # Calculate L2 error
                    error = calculate_l2_error(outputs, predictions)
                    errors.append(error.item())

                    viscosity_value = inputs[0, 2, 0, 0].item()  # Assume all values in channel 2 are the same
                    viscosity_value = viscosity_value * (max_viscosity-min_viscosity) + min_viscosity
                    diffusivity_value = inputs[0, 3, 0, 0].item()  # Assume all values in channel 3 are the same
                    diffusivity_value = diffusivity_value * (max_diffusivity-min_diffusivity) + min_diffusivity
                    
                    viscosity_values.append(viscosity_value)
                    diffusivity_values.append(diffusivity_value)
                    
                    writer.writerow([i, viscosity_value, diffusivity_value, error.item()])
    
    # Plotting the errors against attributes
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(viscosity_values, diffusivity_values, c=errors, cmap='viridis')
    plt.colorbar(sc, label='Relative L2 Error')
    plt.title(f'{model_type} Error vs. Viscosity and Diffusivity')
    plt.xlabel('Viscosity')
    plt.ylabel('Diffusivity')
    plt.grid(True)
    plt.show()
    plt.savefig(f'{model_type}_Viscosity_Diffusivity_to_Error_Plot.png')

# Configure the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_new/model.pkl'
# model_path = 'TrainedModels/FNO_straka_bubble_dt_60_normalized_everywhere/model.pkl'
model_type = model_path.split('TrainedModels/')[1][:3]
model = torch.load(model_path, map_location=device)
model.eval()
autoreg = True
t_in=0
t_out=900
dt=900

if autoreg:
    test_loader = DataLoader(StrakaBubblePlottingDataset(which="test", training_samples=128, model_type=model_type, t_in=t_in, t_out=t_out), batch_size=1, shuffle=False)
else:
    test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type=model_type, dt=dt, normalize=False), batch_size=1, shuffle=False)

csv_filename = f'{model_type}_Viscosity_Diffusivity_Errors.csv'

# Run the error plotting function
plot_error_by_attributes(test_loader, model, model_type, t_in, t_out, dt, autoreg, csv_filename, device)
