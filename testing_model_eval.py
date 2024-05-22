import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Path to the saved model
model_path = 'model.pkl'
# model_path = 'TrainedModels/CNO_poisson_paper/model.pkl'

# Load the entire model directly
model = torch.load(model_path, map_location=torch.device(device))
model.eval()