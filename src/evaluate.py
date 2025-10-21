"""
Evaluation Script for Office Item Classification
Evaluates trained model on test set and generates comprehensive metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
from tqdm import tqdm
import pandas as pd

# Configuration
DATA_DIR = Path('data/processed')
MODEL_PATH = Path('models/best_model.pth')
RESULTS_DIR = Path('results')
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (must match training order)
CLASS_NAMES = [
    'computer_mouse', 'keyboard', 'laptop', 'mobile_phone', 'monitor',
    'notebook', 'office_chair', 'pen', 'water_bottle'
]

def get_test_transforms():
    """
    Define test transforms (no augmentation)
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def create_test_dataloader():
    """
    Create test dataloader
    """
    test_transforms = get_test_transforms()
    
    test_dataset = datasets.ImageFolder(
        DATA_DIR / 'test',
        transform=test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    return test_loader, test_dataset

def load_model(num_classes=9):
    """
    Load trained model
    """
    # Create model architecture (same as training)
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    return model

def evaluate_model(model, test_loader):
    """
    Run inference on test set and collect predictions
    """
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nRunning evaluation on test set...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro F1 score (average across all classes)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(CLASS_NAMES))
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'f1_scores': f1,
        'support': support,
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Create and save confusion matrix visualization
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Predictions'}
    )
    
    plt.title('Confusion Matrix - Office Item Classification\nTest Set Performance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_path}")
    plt.close()

def save_results(metrics, save_dir):
    """
    Save all evaluation results
    """
    save_dir.mkdir(exist_ok=True)
    
    # Save metrics as JSON
    metrics_dict = {
        'accuracy': float(metrics['accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'per_class_metrics': {
            class_name: {
                'precision': float(metrics['precision'][i]),
                'recall': float(metrics['recall'][i]),
                'f1_score': float(metrics['f1_scores'][i]),
                'support': int(metrics['support'][i])
            }
            for i, class_name in enumerate(CLASS_NAMES)
        }
    }
    
    with open(save_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {save_dir / 'test_metrics.json'}")
    
    # Save classification report
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write("OFFICE ITEM CLASSIFICATION - TEST SET EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n\n")
        f.write("Per-Class Performance:\n")
        f.write("-" * 60 + "\n")
        f.write(metrics['classification_report'])
    print(f"Classification report saved to {save_dir / 'classification_report.txt'}")
    
    # Save per-class metrics as CSV
    per_class_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_scores'],
        'Support': metrics['support']
    })
    per_class_df.to_csv(save_dir / 'per_class_metrics.csv', index=False)
    print(f"Per-class metrics saved to {save_dir / 'per_class_metrics.csv'}")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )
    cm_df.to_csv(save_dir / 'confusion_matrix.csv')
    print(f"Confusion matrix saved to {save_dir / 'confusion_matrix.csv'}")

def print_summary(metrics):
    """
    Print evaluation summary to console
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print("\nPer-Class Performance:")
    print("-"*60)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<20} "
              f"{metrics['precision'][i]:<12.4f} "
              f"{metrics['recall'][i]:<12.4f} "
              f"{metrics['f1_scores'][i]:<12.4f}")
    
    print("="*60)
    
    # Find best and worst performing classes
    best_idx = np.argmax(metrics['f1_scores'])
    worst_idx = np.argmin(metrics['f1_scores'])
    
    print(f"\nBest performing class: {CLASS_NAMES[best_idx]} "
          f"(F1: {metrics['f1_scores'][best_idx]:.4f})")
    print(f"Worst performing class: {CLASS_NAMES[worst_idx]} "
          f"(F1: {metrics['f1_scores'][worst_idx]:.4f})")
    print()

def main():
    """
    Main evaluation function
    """
    print("="*60)
    print("OFFICE ITEM CLASSIFICATION - MODEL EVALUATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test Data: {DATA_DIR / 'test'}")
    print("="*60)
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load test data
    print("\nLoading test dataset...")
    test_loader, test_dataset = create_test_dataloader()
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {len(CLASS_NAMES)}")
    
    # Load model
    print("\nLoading trained model...")
    model = load_model(num_classes=len(CLASS_NAMES))
    print("Model loaded successfully!")
    
    # Evaluate
    predictions, labels, probabilities = evaluate_model(model, test_loader)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(labels, predictions)
    
    # Print summary
    print_summary(metrics)
    
    # Plot confusion matrix
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        CLASS_NAMES,
        RESULTS_DIR / 'confusion_matrix.png'
    )
    
    # Save all results
    print("\nSaving results...")
    save_results(metrics, RESULTS_DIR)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Files generated:")
    print("  - test_metrics.json")
    print("  - classification_report.txt")
    print("  - per_class_metrics.csv")
    print("  - confusion_matrix.csv")
    print("  - confusion_matrix.png")
    print("\nNext steps:")
    print("  1. Review confusion matrix for common misclassifications")
    print("  2. Analyze per-class performance")
    print("  3. Create inference.py for single image predictions")

if __name__ == '__main__':
    main()