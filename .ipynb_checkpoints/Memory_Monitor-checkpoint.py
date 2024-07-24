import torch
from torch.utils.data import Dataset, DataLoader
import psutil
import os

class SimpleDataset(Dataset):
    def __init__(self, size):
        self.data = torch.arange(size).view(-1, 1).float()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Create a simple dataset and DataLoader
dataset = SimpleDataset(size=10000)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# Test DataLoader iteration
for i, batch in enumerate(dataloader):
    print(f"Batch {i}: {batch}")
    print_memory_usage()

print("Finished iterating over DataLoader.")
