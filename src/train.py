"""
Training Script for Office Item Classification
Uses PyTorch with transfer learning (ResNet18)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time
from tqdm import tqdm

# Configuration
DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (in alphabetical order as torchvision.datasets.ImageFolder loads them)
CLASS_NAMES = [
    'computer_mouse', 'keyboard', 'laptop', 'mobile_phone', 'mug',
    'notebook', 'office_bin', 'office_chair', 'pen', 'stapler', 'water_bottle'
]

def get_data_transforms():
    """
    Define data augmentation and normalization for training and validation
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
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
    return data_transforms

def create_dataloaders():
    """
    Create training and validation dataloaders
    """
    data_transforms = get_data_transforms()
    
    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(DATA_DIR / 'train', data_transforms['train']),
        'val': datasets.ImageFolder(DATA_DIR / 'val', data_transforms['val'])
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=NUM_WORKERS),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    return dataloaders, dataset_sizes, image_datasets['train'].classes

def create_model(num_classes):
    """
    Create ResNet18 model with transfer learning
    """
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers (optional - can fine-tune all layers)
    for param in model.parameters():
        param.requires_grad = True  # Fine-tune all layers
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    """
    Train the model with validation
    """
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()}')
            
            # Iterate over data
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save to history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Deep copy the model if best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # Save best model
                torch.save(model.state_dict(), MODEL_DIR / 'best_model.pth')
                print(f'New best model saved! Val Acc: {best_acc:.4f}')
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def main():
    """
    Main training function
    """
    print("="*60)
    print("OFFICE ITEM CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("="*60)
    
    # Create models directory
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    dataloaders, dataset_sizes, class_names = create_dataloaders()
    num_classes = len(class_names)
    
    print(f"Classes: {num_classes}")
    print(f"Training samples: {dataset_sizes['train']}")
    print(f"Validation samples: {dataset_sizes['val']}")
    print(f"\nClass names: {class_names}")
    
    # Create model
    print("\nCreating model (ResNet18 with pretrained weights)...")
    model = create_model(num_classes)
    model = model.to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(
        model, dataloaders, dataset_sizes, 
        criterion, optimizer, num_epochs=NUM_EPOCHS
    )
    
    # Save final model
    torch.save(model.state_dict(), MODEL_DIR / 'final_model.pth')
    print(f"\nFinal model saved to {MODEL_DIR / 'final_model.pth'}")
    
    # Save training history
    import json
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        # Convert numpy floats to regular floats for JSON
        history_json = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(history_json, f, indent=2)
    print(f"Training history saved to {MODEL_DIR / 'training_history.json'}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python src/evaluate.py to evaluate on test set")
    print("  2. Run: python src/inference.py <image_path> to classify images")

if __name__ == '__main__':
    main()