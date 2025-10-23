"""
Real-Time Camera Inference for Office Item Classification
Captures live video from webcam and classifies objects in real-time
IMPROVED: Better confidence handling, center crop, warnings
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
CAMERA_INDEX = 0 #change this to your camera of choice
CONFIDENCE_THRESHOLD = 0.80  # Only show predictions above 80%
CENTER_CROP_SCALE = 0.7  # Use center 70% of frame (zoom effect)

# Class names (must match training order)
CLASS_NAMES = [
    'computer_mouse', 'keyboard', 'laptop', 'mobile_phone', 'monitor',
    'notebook', 'office_chair', 'pen', 'water_bottle'
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

def load_model(num_classes=9):
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

def crop_center(frame, crop_scale=CENTER_CROP_SCALE):
    """
    Crop to center region of frame (zoom effect)
    """
    h, w = frame.shape[:2]
    crop_h, crop_w = int(h * crop_scale), int(w * crop_scale)
    start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
    
    cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
    # Resize back to original size for display
    cropped = cv2.resize(cropped, (w, h))
    
    return cropped

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

def draw_prediction_overlay(frame, predicted_class, confidence, top3_predictions, fps, use_crop):
    """
    Draw prediction information on the frame with improved confidence handling
    """
    height, width = frame.shape[:2]
    
    # Draw center crop guide box if cropping is enabled
    if use_crop:
        crop_h, crop_w = int(height * CENTER_CROP_SCALE), int(width * CENTER_CROP_SCALE)
        start_y, start_x = (height - crop_h) // 2, (width - crop_w) // 2
        cv2.rectangle(frame, (start_x, start_y), 
                     (start_x + crop_w, start_y + crop_h), 
                     (0, 255, 255), 2)  # Yellow guide box
        cv2.putText(frame, "DETECTION AREA", (start_x + 10, start_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Create semi-transparent overlay panel
    overlay = frame.copy()
    panel_height = 180
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title
    cv2.putText(frame, "Office Item Classifier", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Confidence-based display
    if confidence >= CONFIDENCE_THRESHOLD:
        # HIGH CONFIDENCE - Show prediction
        formatted_class = format_class_name(predicted_class)
        conf_color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
        
        cv2.putText(frame, f"Detected: {formatted_class}", (10, 75),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, conf_color, 2)
        
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        
        # Confidence bar
        bar_width = int(confidence * 400)
        bar_x = 200
        bar_y = 100
        bar_height = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 400, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), conf_color, -1)
        
    else:
        # LOW CONFIDENCE - Show warning
        cv2.putText(frame, "LOW CONFIDENCE - Adjusting...", (10, 75),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 165, 255), 2)
        
        cv2.putText(frame, f"Best guess: {format_class_name(predicted_class)} ({confidence*100:.1f}%)", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show helpful tips
        cv2.putText(frame, ">> Move object CLOSER to camera", (10, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Bottom instructions
    cv2.putText(frame, "TIP: Center object in yellow box | Plain background helps", 
                (10, height - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'q' to quit | 's' to save | 'c' to toggle crop zoom", 
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def main():
    """
    Main camera inference loop
    """
    print("="*60)
    print("OFFICE ITEM CLASSIFICATION - LIVE CAMERA (IMPROVED)")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = load_model()
    transform = get_transform()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD*100}%")
    
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
    print("\nIMPROVEMENTS:")
    print("  ✓ Center crop zoom mode (press 'c' to toggle)")
    print("  ✓ Confidence threshold (80% minimum)")
    print("  ✓ Visual detection area guide")
    print("  ✓ Low confidence warnings with tips")
    print("\nInstructions:")
    print("  - Center object in yellow detection box")
    print("  - Keep object close to camera (fills 60%+ of box)")
    print("  - Use plain background when possible")
    print("  - Press 'c' to toggle crop zoom")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("\nStarting live inference...\n")
    
    # State variables
    prev_time = time.time()
    fps = 0
    screenshot_count = 0
    use_crop = True  # Start with crop enabled
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Apply center crop if enabled
            display_frame = crop_center(frame) if use_crop else frame
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Run prediction
            predicted_class, confidence, top3_predictions = predict_frame(
                display_frame, model, transform
            )
            
            # Draw overlay
            frame_with_overlay = draw_prediction_overlay(
                display_frame, predicted_class, confidence, top3_predictions, fps, use_crop
            )
            
            # Display
            cv2.imshow('Office Item Classifier - Live Camera (IMPROVED)', frame_with_overlay)
            
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
            elif key == ord('c'):
                use_crop = not use_crop
                mode = "ENABLED" if use_crop else "DISABLED"
                print(f"Crop zoom: {mode}")
    
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