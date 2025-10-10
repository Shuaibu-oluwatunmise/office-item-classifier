"""
Inference Script for Office Item Classification
Classifies a single image file and displays prediction with confidence
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from pathlib import Path

# Configuration
MODEL_PATH = Path('models/best_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (must match training order)
CLASS_NAMES = [
    'computer_mouse', 'keyboard', 'laptop', 'mobile_phone', 'mug',
    'notebook', 'office_bin', 'office_chair', 'pen', 'stapler', 'water_bottle'
]

def get_transform():
    """
    Define image preprocessing transforms (same as test set)
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_model(num_classes=11):
    """
    Load trained model
    """
    # Create model architecture
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    return model

def predict_image(image_path, model, transform):
    """
    Predict class for a single image
    """
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None
    
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_predictions = [
        (CLASS_NAMES[idx.item()], prob.item()) 
        for idx, prob in zip(top3_idx[0], top3_prob[0])
    ]
    
    return predicted_class, confidence_score, top3_predictions

def format_class_name(class_name):
    """
    Format class name for display (replace underscores with spaces, capitalize)
    """
    return ' '.join(word.capitalize() for word in class_name.split('_'))

def print_prediction(image_path, predicted_class, confidence, top3_predictions):
    """
    Print prediction results in a nice format
    """
    print("\n" + "="*60)
    print("OFFICE ITEM CLASSIFICATION - PREDICTION")
    print("="*60)
    print(f"Image: {image_path}")
    print("-"*60)
    print(f"\n🎯 Predicted Class: {format_class_name(predicted_class)}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print("\n" + "-"*60)
    print("Top 3 Predictions:")
    print("-"*60)
    
    for i, (cls, prob) in enumerate(top3_predictions, 1):
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{i}. {format_class_name(cls):20s} {bar} {prob*100:5.2f}%")
    
    print("="*60 + "\n")

def main():
    """
    Main inference function
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python src/inference.py <path_to_image>")
        print("\nExample:")
        print("  python src/inference.py data/processed/test/mug/image.jpg")
        print("  python src/inference.py my_office_photo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print("\nLoading model...")
    
    # Load model
    model = load_model()
    transform = get_transform()
    
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Run prediction
    print(f"\nAnalyzing image: {image_path}")
    predicted_class, confidence, top3_predictions = predict_image(
        image_path, model, transform
    )
    
    if predicted_class is None:
        print("Failed to process image.")
        sys.exit(1)
    
    # Display results
    print_prediction(image_path, predicted_class, confidence, top3_predictions)
    
    # Confidence warning
    if confidence < 0.7:
        print("⚠️  Warning: Low confidence prediction (< 70%)")
        print("   The image may be ambiguous or not clearly show the object.")
        print("   Consider using a clearer image or different angle.\n")
    elif confidence > 0.95:
        print("✅ High confidence prediction! The model is very certain.\n")

if __name__ == '__main__':
    main()