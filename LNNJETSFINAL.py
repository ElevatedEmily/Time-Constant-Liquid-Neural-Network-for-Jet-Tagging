import os
import numpy as np
import awkward as ak
import uproot
import vector
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import torch.cuda.amp as amp
import matplotlib.pyplot as plt

vector.register_awkward()

# Define the function to read the ROOT files
def read_file(filepath, max_num_particles=128):
    """Loads a single file from the JetClass dataset."""
    
    table = uproot.open(filepath)['tree'].arrays()

    p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})

    # Select px, py, pz as features
    particle_features = ['part_px', 'part_py', 'part_pz']
    x_particles = np.stack([ak.to_numpy(ak.fill_none(ak.pad_none(table[n], max_num_particles, clip=True), 0)) for n in particle_features], axis=0)

    # Assuming that your labels are stored in a single variable and one-hot encoded
    labels = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
              'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=0)
    y = np.any(y, axis=0).astype(int)  # Ensure only one-hot encoded label
    
    return x_particles, y

# Define the dataset class
class JetClassDataset(Dataset):
    def __init__(self, filepaths, max_num_particles=128):
        self.filepaths = filepaths
        self.max_num_particles = max_num_particles

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        x_particles, y = read_file(filepath, max_num_particles=self.max_num_particles)

        # Convert to tensors
        x_particles = torch.tensor(x_particles, dtype=torch.float32)  # Shape: [3, max_num_particles]
        y = torch.tensor(y, dtype=torch.float32)  # Shape: [num_classes]

        return x_particles, y

# Directory to save checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, scaler, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print(f"Checkpoint saved at epoch {epoch + 1}")

# Function to load the model checkpoint
def load_checkpoint(filename="checkpoint.pth"):
    checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {start_epoch + 1}")
    return start_epoch

# Function to gather file paths
def get_filepaths_from_dir(directory):
    filepaths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".root"):
                filepaths.append(os.path.join(root, file))
    return filepaths

# Example directories (update with actual paths)
train_dir = "G:/PCN-Jet-Tagging-master/data/train"
val_dir = "G:/PCN-Jet-Tagging-master/data/val"
test_dir = "G:/PCN-Jet-Tagging-master/data/test"

# Gather file paths
train_filepaths = get_filepaths_from_dir(train_dir)
val_filepaths = get_filepaths_from_dir(val_dir)
test_filepaths = get_filepaths_from_dir(test_dir)

# Ensure there are files to process
if not train_filepaths or not val_filepaths or not test_filepaths:
    raise ValueError("No .root files found in the specified directories.")

# Create datasets
train_dataset = JetClassDataset(train_filepaths)
val_dataset = JetClassDataset(val_filepaths)
test_dataset = JetClassDataset(test_filepaths)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

# Define the Liquid Neural Network with 3D convolutions
class LiquidNeuralNetwork3D(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(LiquidNeuralNetwork3D, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool3d((8, 8, 8))  # Example pooling, adjust as needed
        self.fc1 = nn.Linear(32 * 8 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension, shape: [batch_size, 1, 3, max_num_particles, 1]
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the network, loss function, and optimizer
input_channels = 1  # For px, py, pz combined
output_dim = 10  # Number of classes
model = LiquidNeuralNetwork3D(input_channels, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mixed precision training setup
scaler = amp.GradScaler()

def calculate_accuracy(outputs, targets):
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == targets).sum().item()
    return correct / targets.size(0)


# Resume training setup
resume_training = False  # Set to True if you want to resume training
start_epoch = 0

if resume_training:
    start_epoch = load_checkpoint(filename="checkpoint.pth")

# Training loop with gradient accumulation and checkpointing
num_epochs = 50
accumulation_steps = 8  # Adjust based on available memory and desired batch size
patience = 5
best_val_loss = float('inf')
early_stop_counter = 0

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(start_epoch, num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Convert one-hot encoded targets to class indices
        targets = torch.argmax(targets, dim=1)
        
        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # No need to divide by accumulation_steps here
            
        scaler.scale(loss).backward()

        # Perform optimization step after accumulating gradients
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Reset gradients for the next accumulation cycle

        running_loss += loss.item() * inputs.size(0)
        running_corrects += calculate_accuracy(outputs, targets) * inputs.size(0)
    
    # If there are leftover gradients after the loop, update weights
    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation phase
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Convert one-hot encoded targets to class indices
            targets = torch.argmax(targets, dim=1)
            
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += calculate_accuracy(outputs, targets) * inputs.size(0)
    
    val_loss = running_loss / len(val_dataset)
    val_acc = running_corrects / len(val_dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

    # Save a checkpoint after every epoch
    save_checkpoint(model, optimizer, scaler, epoch, filename="checkpoint.pth")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # Save the best model
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

print("Finished Training")

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Evaluate the model on test set
test_loss = 0.0
test_corrects = 0
model.eval()
with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Testing"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Convert one-hot encoded targets to class indices
        targets = torch.argmax(targets, dim=1)
        
        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        test_loss += loss.item() * inputs.size(0)
        test_corrects += calculate_accuracy(outputs, targets) * inputs.size(0)

test_loss /= len(test_dataset)
test_acc = test_corrects / len(test_dataset)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Plot training and validation loss and accuracy
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()
