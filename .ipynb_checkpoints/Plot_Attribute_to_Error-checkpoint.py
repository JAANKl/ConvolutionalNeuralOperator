import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Assuming StrakaBubbleDataset and necessary imports are correctly configured
from Dataloaders import StrakaBubbleDataset

min_viscosity = 0.029050784477663294
max_viscosity = 74.91834790806836
min_diffusivity = 0.043989764422611155
max_diffusivity = 74.8587964361266

def calculate_l2_error(output, prediction):
    return torch.norm(output - prediction, p=2) / torch.norm(output, p=2)

def plot_error_by_diffusivity(data_loader, model, attribute, model_type, device):
    attribute_values = []
    errors = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            for inputs, outputs in batch:
                inputs, outputs = inputs.to(device), outputs.to(device)

                # Make predictions
                predictions = model(inputs)
                
                if model_type == "FNO":
                    inputs = inputs.permute(0,3,1,2)
                    outputs = outputs.permute(0,3,1,2)
                    predictions = predictions.permute(0,3,1,2)

                # Calculate L2 error
                error = calculate_l2_error(outputs, predictions)
                errors.append(error.item())

                # Get diffusivity value (assuming it's constant and the same for all points in a batch)
                if attribute == "Diffusivity":
                    attribute_value = inputs[0, 3, 0, 0].item()  # Assume all values in channel 3 are the same
                    attribute_value = attribute_value * (max_diffusivity-min_diffusivity) + min_diffusivity
                elif attribute == "Viscosity":
                    attribute_value = inputs[0, 2, 0, 0].item()  # Assume all values in channel 2 are the same
                    attribute_value = attribute_value * (max_viscosity-min_viscosity) + min_viscosity
                else:
                    raise NotImplementedError("Only Viscosity and Diffusivity supported")
                attribute_values.append(attribute_value)
    
    # Plotting the errors against diffusivity
    plt.figure(figsize=(10, 6))
    plt.scatter(attribute_values, errors, color='blue')
    plt.title(model_type + ' L2 Error vs. ' + attribute)
    plt.xlabel(attribute)
    plt.ylabel('L2 Error')
    plt.grid(True)
    plt.show()
    plt.savefig(model_type + "_" + attribute + '_to_error_plot.png')

# Configure the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model_path = 'TrainedModels/CNO_straka_bubble_0_to_900_normalized_everywhere/model.pkl'
model_type = model_path.split('TrainedModels/')[1][:3]
attribute = "Diffusivity"
# attribute = "Viscosity"
model = torch.load(model_path, map_location=device)
model.eval()

# Setup the DataLoader
test_loader = DataLoader(StrakaBubbleDataset(which="test", training_samples=128, model_type=model_type, dt=900, normalize=True), batch_size=1)

# Run the error plotting function
plot_error_by_diffusivity(test_loader, model, attribute, model_type, device)
