"""
Multi-Model Inference Script for Office Item Classification
Classifies a single image file using selected model
Supports: ResNet18, YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
CLASS_NAMES = [
    'computer_mouse', 'keyboard', 'laptop', 'mobile_phone', 'monitor',
    'notebook', 'office_chair', 'pen', 'water_bottle'
]

# Model paths
MODEL_PATHS = {
    'resnet': Path('models/best_model.pth'),
    'yolov8n': Path('runs/classify/yolov8n-cls_train/weights/best.pt'),
    'yolov8s': Path('runs/classify/yolov8s-cls_train/weights/best.pt'),
    'yolov11n': Path('runs/classify/yolov11n-cls_train/weights/best.pt'),
    'yolov11s': Path('runs/classify/yolov11s-cls_train/weights/best.pt'),
}

def get_transform():
    """Define image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_resnet_model():
    """Load ResNet18 model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATHS['resnet'], map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def load_yolo_model(model_name):
    """Load YOLO model"""
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))

def predict_resnet(image_path, model, transform):
    """Predict using ResNet"""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    # Get top 3
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_predictions = [
        (CLASS_NAMES[idx.item()], prob.item()) 
        for idx, prob in zip(top3_idx[0], top3_prob[0])
    ]
    
    return predicted_class, confidence_score, top3_predictions

def predict_yolo(image_path, model):
    """Predict using YOLO"""
    try:
        results = model(str(image_path), verbose=False)
        
        # Get prediction
        probs = results[0].probs
        predicted_class_idx = probs.top1
        confidence_score = probs.top1conf.item()
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get top 3
        top3_indices = probs.top5[:3]  # Get top 3 from top 5
        top3_probs = [probs.data[i].item() for i in top3_indices]
        top3_predictions = [
            (CLASS_NAMES[idx], prob) 
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        
        return predicted_class, confidence_score, top3_predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

def format_class_name(class_name):
    """Format class name for display"""
    return ' '.join(word.capitalize() for word in class_name.split('_'))

def print_prediction(image_path, predicted_class, confidence, top3_predictions, model_name):
    """Print prediction results"""
    model_display = {
        'resnet': 'ResNet18',
        'yolov8n': 'YOLOv8n-cls',
        'yolov8s': 'YOLOv8s-cls',
        'yolov11n': 'YOLOv11n-cls',
        'yolov11s': 'YOLOv11s-cls',
    }
    
    print("\n" + "="*70)
    print("OFFICE ITEM CLASSIFICATION - MULTI-MODEL INFERENCE")
    print("="*70)
    print(f"Model: {model_display[model_name]}")
    print(f"Image: {image_path}")
    print("-"*70)
    print(f"\n🎯 Predicted Class: {format_class_name(predicted_class)}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print("\n" + "-"*70)
    print("Top 3 Predictions:")
    print("-"*70)
    
    for i, (cls, prob) in enumerate(top3_predictions, 1):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{i}. {format_class_name(cls):20s} {bar} {prob*100:5.2f}%")
    
    print("="*70 + "\n")
    
    # Confidence feedback
    if confidence < 0.7:
        print("⚠️  Warning: Low confidence prediction (< 70%)")
        print("   Consider using a clearer image or different angle.\n")
    elif confidence > 0.95:
        print("✅ High confidence prediction! The model is very certain.\n")

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description='Office Item Classification - Multi-Model Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inference.py image.jpg --model resnet
  python src/inference.py image.jpg --model yolov8n
  python src/inference.py image.jpg --model yolov8s
  python src/inference.py image.jpg --model yolov11n
  python src/inference.py image.jpg --model yolov11s
        """
    )
    
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet', 'yolov8n', 'yolov8s', 'yolov11n', 'yolov11s'],
                       help='Model to use for inference')
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Check if model exists
    if not MODEL_PATHS[args.model].exists():
        print(f"❌ Error: Model not found: {MODEL_PATHS[args.model]}")
        print(f"   Make sure the model is trained and saved.")
        sys.exit(1)
    
    print(f"\n🔄 Loading {args.model.upper()} model...")
    
    # Load model and run inference
    try:
        if args.model == 'resnet':
            model = load_resnet_model()
            transform = get_transform()
            print(f"✅ Model loaded: {MODEL_PATHS['resnet']}")
            print(f"📍 Device: {DEVICE}\n")
            print(f"🔍 Analyzing image...")
            predicted_class, confidence, top3_predictions = predict_resnet(
                image_path, model, transform
            )
        else:
            model = load_yolo_model(args.model)
            print(f"✅ Model loaded: {MODEL_PATHS[args.model]}")
            print(f"📍 Device: {DEVICE}\n")
            print(f"🔍 Analyzing image...")
            predicted_class, confidence, top3_predictions = predict_yolo(
                image_path, model
            )
        
        if predicted_class is None:
            print("❌ Failed to process image.")
            sys.exit(1)
        
        # Display results
        print_prediction(image_path, predicted_class, confidence, 
                        top3_predictions, args.model)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()