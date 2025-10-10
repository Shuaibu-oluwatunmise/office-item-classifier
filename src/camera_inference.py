"""
Real-Time Camera Inference for Office Item Classification
Captures live video from webcam and classifies objects in real-time
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import time

# Configuration
MODEL_PATH = Path('models/best_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CAMERA_INDEX = 0  # Change to 1 or 2 if default camera doesn't work

# Class names (must match training order)
CLASS_NAMES = [
    'computer_mouse', 'keyboard', 'laptop', 'mobile_phone', 'mug',
    'notebook', 'office_bin', 'office_chair', 'pen', 'stapler', 'water_bottle'
]

def get_transform():
    """
    Define image preprocessing transforms
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_model(num_classes=11):
    """
    Load trained model
    """
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    return model

def format_class_name(class_name):
    """
    Format class name for display
    """
    return ' '.join(word.capitalize() for word in class_name.split('_'))

def predict_frame(frame, model, transform):
    """
    Predict class for a video frame
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Transform
    input_tensor = transform(rgb_frame).unsqueeze(0).to(DEVICE)
    
    # Predict
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

def draw_prediction_overlay(frame, predicted_class, confidence, top3_predictions, fps):
    """
    Draw prediction information on the frame (simplified - main prediction only)
    """
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay panel (smaller now)
    overlay = frame.copy()
    panel_height = 140
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title
    cv2.putText(frame, "Office Item Classifier", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Main prediction (larger and more prominent)
    formatted_class = format_class_name(predicted_class)
    cv2.putText(frame, f"Detected: {formatted_class}", (10, 75),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
    
    # Confidence with visual bar
    conf_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
    
    # Confidence bar
    bar_width = int(confidence * 400)
    bar_x = 200
    bar_y = 95
    bar_height = 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 400, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), conf_color, -1)
    
    # Instructions and tip
    cv2.putText(frame, "TIP: Hold object close with plain background", 
                (10, height - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'q' to quit | Press 's' to save screenshot", 
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def main():
    """
    Main camera inference loop
    """
    print("="*60)
    print("OFFICE ITEM CLASSIFICATION - LIVE CAMERA")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = load_model()
    transform = get_transform()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Initialize camera
    print(f"\nInitializing camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected and not used by another app")
        print("  2. Try changing CAMERA_INDEX in the script (0, 1, or 2)")
        print("  3. Grant camera permissions if prompted")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera initialized successfully!")
    print("\nInstructions:")
    print("  - Point camera at office items")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("\nStarting live inference...\n")
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    screenshot_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Run prediction
            predicted_class, confidence, top3_predictions = predict_frame(
                frame, model, transform
            )
            
            # Draw overlay
            frame_with_overlay = draw_prediction_overlay(
                frame, predicted_class, confidence, top3_predictions, fps
            )
            
            # Display
            cv2.imshow('Office Item Classifier - Live Camera', frame_with_overlay)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame_with_overlay)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Goodbye!")
        print("="*60)

if __name__ == '__main__':
    main()