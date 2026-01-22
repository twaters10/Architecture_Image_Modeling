import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. Setup Hyperparameters
DATA_DIR = 'data' # Path to your train/val folders
BATCH_SIZE = 32
NUM_CLASSES = 10
LEARNING_RATE = 0.001
EPOCHS = 10
# 1. Check for MPS device availability
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple GPU with Metal backend.")
else:
    DEVICE = torch.device("cpu")
    print("MPS device not found, falling back to CPU.")

# 2. Data Augmentation and Normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. Load Datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}

# Print the mapping of folder names to class IDs
print(f"Detected Classes: {image_datasets['train'].classes}")
print(f"Class to ID Mapping: {image_datasets['train'].class_to_idx}")

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 4. Load Pretrained Model & Fine-tune
model = models.resnet18(weights='DEFAULT')

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer (the "head")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# 5. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# Only optimize the parameters of the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# 6. Training Loop
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# Run the training
model_ft = train_model(model, criterion, optimizer, num_epochs=EPOCHS)

# 7. Save the model
torch.save(model_ft.state_dict(), 'house_style_model.pth')