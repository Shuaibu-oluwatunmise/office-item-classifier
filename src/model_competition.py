"""
Ultimate Model Competition - Office Item Classification
Tests all 5 models on excesses folder and crowns the champion!
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Configuration
ROOT_DIR = Path(__file__).parent.parent.absolute()
EXCESSES_DIR = ROOT_DIR / "excesses"
RESULTS_DIR = ROOT_DIR / "results" / "competition"
DEVICE = torch.device('cpu')  # Force CPU

CLASS_NAMES = [
    'computer_mouse', 'keyboard', 'laptop', 'mobile_phone', 'monitor',
    'notebook', 'office_chair', 'pen', 'water_bottle'
]

# Model paths
MODELS = {
    "ResNet18": {
        "type": "pytorch",
        "path": ROOT_DIR / "models" / "best_model.pth"
    },
    "YOLOv8n": {
        "type": "yolo",
        "path": ROOT_DIR / "runs" / "classify" / "yolov8n-cls_train" / "weights" / "best.pt"
    },
    "YOLOv8s": {
        "type": "yolo",
        "path": ROOT_DIR / "runs" / "classify" / "yolov8s-cls_train" / "weights" / "best.pt"
    },
    "YOLOv11n": {
        "type": "yolo",
        "path": ROOT_DIR / "runs" / "classify" / "yolov11n-cls_train" / "weights" / "best.pt"
    },
    "YOLOv11s": {
        "type": "yolo",
        "path": ROOT_DIR / "runs" / "classify" / "yolov11s-cls_train" / "weights" / "best.pt"
    }
}

def get_transforms():
    """Test transforms"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_resnet(model_path):
    """Load ResNet model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def evaluate_resnet(model, data_loader):
    """Evaluate ResNet model"""
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating ResNet"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_confidences)

def evaluate_yolo(model_path, data_dir):
    """Evaluate YOLO model"""
    model = YOLO(str(model_path))
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    # Get all class folders
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    for class_idx, class_folder in enumerate(tqdm(class_folders, desc=f"Evaluating YOLO")):
        # Get all images in this class
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        for img_path in image_files:
            # Predict
            results = model(str(img_path), verbose=False)
            
            # Get prediction
            pred_class = results[0].probs.top1
            confidence = results[0].probs.top1conf.item()
            
            all_preds.append(pred_class)
            all_labels.append(class_idx)
            all_confidences.append(confidence)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_confidences)

def calculate_metrics(y_true, y_pred, confidences):
    """Calculate comprehensive metrics"""
    accuracy = (y_true == y_pred).mean()
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(len(CLASS_NAMES)):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Average confidence
    avg_confidence = confidences.mean()
    correct_confidence = confidences[y_true == y_pred].mean() if (y_true == y_pred).sum() > 0 else 0
    incorrect_confidence = confidences[y_true != y_pred].mean() if (y_true != y_pred).sum() > 0 else 0
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'avg_confidence': avg_confidence,
        'correct_confidence': correct_confidence,
        'incorrect_confidence': incorrect_confidence,
        'total_samples': len(y_true)
    }

def plot_comparison(all_results, save_dir):
    """Create comprehensive comparison visualizations"""
    
    # 1. Overall Accuracy Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    models = list(all_results.keys())
    accuracies = [all_results[m]['accuracy'] * 100 for m in models]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = axes[0, 0].bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([95, 100])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confidence comparison
    avg_confs = [all_results[m]['avg_confidence'] * 100 for m in models]
    axes[0, 1].bar(models, avg_confs, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Average Confidence', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([90, 100])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for bar, conf in zip(axes[0, 1].patches, avg_confs):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{conf:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Per-class heatmap
    per_class_data = []
    for model in models:
        per_class_data.append([acc * 100 for acc in all_results[model]['per_class_accuracy']])
    
    sns.heatmap(per_class_data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=models,
                cbar_kws={'label': 'Accuracy (%)'}, ax=axes[1, 0],
                vmin=95, vmax=100)
    axes[1, 0].set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # 4. Correct vs Incorrect Confidence
    x = np.arange(len(models))
    width = 0.35
    
    correct_confs = [all_results[m]['correct_confidence'] * 100 for m in models]
    incorrect_confs = [all_results[m]['incorrect_confidence'] * 100 for m in models]
    
    axes[1, 1].bar(x - width/2, correct_confs, width, label='Correct Predictions',
                   color='#2ecc71', edgecolor='black')
    axes[1, 1].bar(x + width/2, incorrect_confs, width, label='Incorrect Predictions',
                   color='#e74c3c', edgecolor='black')
    
    axes[1, 1].set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Confidence: Correct vs Incorrect Predictions',
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_competition_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Competition chart saved: {save_dir / 'model_competition_results.png'}")
    plt.close()

def create_leaderboard(all_results, save_dir):
    """Create detailed leaderboard"""
    
    # Prepare data
    leaderboard_data = []
    for model_name, results in all_results.items():
        leaderboard_data.append({
            'Model': model_name,
            'Accuracy (%)': results['accuracy'] * 100,
            'Avg Confidence (%)': results['avg_confidence'] * 100,
            'Correct Conf (%)': results['correct_confidence'] * 100,
            'Samples': results['total_samples']
        })
    
    # Sort by accuracy
    leaderboard_data.sort(key=lambda x: x['Accuracy (%)'], reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(leaderboard_data)
    
    # Add ranking
    df.insert(0, 'Rank', ['🥇', '🥈', '🥉', '4th', '5th'][:len(df)])
    
    # Save as CSV
    df.to_csv(save_dir / 'leaderboard.csv', index=False)
    
    # Create formatted text report
    with open(save_dir / 'competition_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("🏆 ULTIMATE MODEL COMPETITION - OFFICE ITEM CLASSIFIER 🏆\n")
        f.write("="*80 + "\n")
        f.write(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Set: Excesses Folder\n")
        f.write(f"Total Samples: {leaderboard_data[0]['Samples']}\n")
        f.write(f"Classes: {len(CLASS_NAMES)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("FINAL LEADERBOARD\n")
        f.write("="*80 + "\n\n")
        
        for i, row in df.iterrows():
            f.write(f"{row['Rank']} {row['Model']}\n")
            f.write(f"   Accuracy:         {row['Accuracy (%)']:.2f}%\n")
            f.write(f"   Avg Confidence:   {row['Avg Confidence (%)']:.2f}%\n")
            f.write(f"   Correct Conf:     {row['Correct Conf (%)']:.2f}%\n")
            f.write("\n")
        
        # Winner analysis
        winner = leaderboard_data[0]
        f.write("="*80 + "\n")
        f.write("🎉 CHAMPION 🎉\n")
        f.write("="*80 + "\n")
        f.write(f"\n{winner['Model']} wins with {winner['Accuracy (%)']:.2f}% accuracy!\n")
        f.write(f"Average confidence: {winner['Avg Confidence (%)']:.2f}%\n")
        
    print(f"\n📋 Leaderboard saved: {save_dir / 'leaderboard.csv'}")
    print(f"📄 Report saved: {save_dir / 'competition_report.txt'}")
    
    return df

def main():
    """Main competition function"""
    print("\n" + "="*80)
    print("🏆 ULTIMATE MODEL COMPETITION 🏆")
    print("Office Item Classification Championship")
    print("="*80)
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    print(f"\n📂 Loading test data from: {EXCESSES_DIR}")
    test_transform = get_transforms()
    test_dataset = datasets.ImageFolder(EXCESSES_DIR, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16)
    
    print(f"✅ Loaded {len(test_dataset)} images across {len(CLASS_NAMES)} classes\n")
    
    # Evaluate all models
    all_results = {}
    
    for model_name, model_info in MODELS.items():
        print(f"\n{'='*80}")
        print(f"🤖 Evaluating: {model_name}")
        print(f"{'='*80}")
        
        if not model_info['path'].exists():
            print(f"⚠️  Model not found: {model_info['path']}")
            print(f"   Skipping {model_name}...")
            continue
        
        try:
            if model_info['type'] == 'pytorch':
                model = load_resnet(model_info['path'])
                preds, labels, confidences = evaluate_resnet(model, test_loader)
            else:  # YOLO
                preds, labels, confidences = evaluate_yolo(model_info['path'], EXCESSES_DIR)
            
            # Calculate metrics
            metrics = calculate_metrics(labels, preds, confidences)
            all_results[model_name] = metrics
            
            print(f"\n✅ {model_name} Results:")
            print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"   Avg Confidence: {metrics['avg_confidence']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    if not all_results:
        print("\n❌ No models were successfully evaluated!")
        return
    
    # Save detailed results
    with open(RESULTS_DIR / 'detailed_results.json', 'w') as f:
        results_json = {}
        for model_name, metrics in all_results.items():
            results_json[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'per_class_accuracy': [float(x) for x in metrics['per_class_accuracy']],
                'avg_confidence': float(metrics['avg_confidence']),
                'correct_confidence': float(metrics['correct_confidence']),
                'incorrect_confidence': float(metrics['incorrect_confidence']),
                'total_samples': int(metrics['total_samples'])
            }
        json.dump(results_json, f, indent=2)
    
    # Create visualizations
    print("\n" + "="*80)
    print("📊 Creating Visualizations...")
    print("="*80)
    plot_comparison(all_results, RESULTS_DIR)
    
    # Create leaderboard
    print("\n" + "="*80)
    print("🏆 Generating Leaderboard...")
    print("="*80)
    df = create_leaderboard(all_results, RESULTS_DIR)
    
    # Print final leaderboard
    print("\n" + "="*80)
    print("FINAL LEADERBOARD")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("🎉 COMPETITION COMPLETE! 🎉")
    print("="*80)
    print(f"\n📁 All results saved to: {RESULTS_DIR}/")
    print("\nFiles generated:")
    print("  ✅ model_competition_results.png - Visual comparison")
    print("  ✅ leaderboard.csv - Rankings table")
    print("  ✅ competition_report.txt - Detailed report")
    print("  ✅ detailed_results.json - Raw metrics")

if __name__ == '__main__':
    main()