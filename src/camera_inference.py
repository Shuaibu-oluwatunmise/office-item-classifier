"""
Multi-Model Real-Time Camera Inference for Office Item Classification
Supports: ResNet18, YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CAMERA_INDEX = 1
CONFIDENCE_THRESHOLD = 0.80
CENTER_CROP_SCALE = 0.7

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
        transforms.ToPILImage(),
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
    return model, get_transform()

def load_yolo_model(model_name):
    """Load YOLO model"""
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path)), None

def format_class_name(class_name):
    """Format class name for display"""
    return ' '.join(word.capitalize() for word in class_name.split('_'))

def crop_center(frame, crop_scale=CENTER_CROP_SCALE):
    """Crop to center region of frame"""
    h, w = frame.shape[:2]
    crop_h, crop_w = int(h * crop_scale), int(w * crop_scale)
    start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
    
    cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped = cv2.resize(cropped, (w, h))
    
    return cropped

def predict_frame_resnet(frame, model, transform):
    """Predict using ResNet"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb_frame).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score

def predict_frame_yolo(frame, model):
    """Predict using YOLO"""
    results = model(frame, verbose=False)
    probs = results[0].probs
    predicted_class = CLASS_NAMES[probs.top1]
    confidence_score = probs.top1conf.item()
    
    return predicted_class, confidence_score

def draw_prediction_overlay(frame, predicted_class, confidence, fps, use_crop, model_name):
    """Draw prediction information on frame"""
    height, width = frame.shape[:2]
    
    model_display = {
        'resnet': 'ResNet18',
        'yolov8n': 'YOLOv8n',
        'yolov8s': 'YOLOv8s',
        'yolov11n': 'YOLOv11n',
        'yolov11s': 'YOLOv11s',
    }
    
    # Draw center crop guide box
    if use_crop:
        crop_h, crop_w = int(height * CENTER_CROP_SCALE), int(width * CENTER_CROP_SCALE)
        start_y, start_x = (height - crop_h) // 2, (width - crop_w) // 2
        cv2.rectangle(frame, (start_x, start_y), 
                     (start_x + crop_w, start_y + crop_h), 
                     (0, 255, 255), 2)
        cv2.putText(frame, "DETECTION AREA", (start_x + 10, start_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Create overlay panel
    overlay = frame.copy()
    panel_height = 180
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title with model name
    cv2.putText(frame, f"Office Classifier - {model_display[model_name]}", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Prediction display
    if confidence >= CONFIDENCE_THRESHOLD:
        formatted_class = format_class_name(predicted_class)
        conf_color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
        
        cv2.putText(frame, f"Detected: {formatted_class}", (10, 75),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, conf_color, 2)
        
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        
        # Confidence bar
        bar_width = int(confidence * 400)
        bar_x, bar_y, bar_height = 200, 100, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 400, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), conf_color, -1)
    else:
        cv2.putText(frame, "LOW CONFIDENCE - Adjusting...", (10, 75),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 165, 255), 2)
        
        cv2.putText(frame, f"Best guess: {format_class_name(predicted_class)} ({confidence*100:.1f}%)", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, ">> Move object CLOSER to camera", (10, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Instructions
    cv2.putText(frame, "TIP: Center object in yellow box | Plain background helps", 
                (10, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'q' to quit | 's' to save | 'c' to toggle crop", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def select_model():
    """Interactive model selection"""
    print("\n" + "="*70)
    print("SELECT MODEL FOR CAMERA INFERENCE")
    print("="*70)
    print("\nAvailable models:")
    print("  1. ResNet18       (99.85% test accuracy)")
    print("  2. YOLOv8n-cls    (100% test accuracy, fast)")
    print("  3. YOLOv8s-cls    (100% test accuracy, 🏆 CHAMPION)")
    print("  4. YOLOv11n-cls   (100% test accuracy, fast)")
    print("  5. YOLOv11s-cls   (100% test accuracy)")
    print("-"*70)
    
    model_map = {
        '1': 'resnet',
        '2': 'yolov8n',
        '3': 'yolov8s',
        '4': 'yolov11n',
        '5': 'yolov11s',
    }
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice in model_map:
            model_name = model_map[choice]
            
            # Check if model exists
            if not MODEL_PATHS[model_name].exists():
                print(f"❌ Error: Model not found: {MODEL_PATHS[model_name]}")
                print("   Please train the model first or choose another.")
                continue
            
            return model_name
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")

def main():
    """Main camera inference loop"""
    print("="*70)
    print("MULTI-MODEL CAMERA INFERENCE - OFFICE ITEM CLASSIFICATION")
    print("="*70)
    
    # Select model
    model_name = select_model()
    
    # Load model
    print(f"\n🔄 Loading {model_name.upper()} model...")
    try:
        if model_name == 'resnet':
            model, transform = load_resnet_model()
            predict_fn = lambda frame: predict_frame_resnet(frame, model, transform)
        else:
            model, _ = load_yolo_model(model_name)
            predict_fn = lambda frame: predict_frame_yolo(frame, model)
        
        print(f"✅ Model loaded: {MODEL_PATHS[model_name]}")
        print(f"📍 Device: {DEVICE}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize camera
    print(f"\n📷 Initializing camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {CAMERA_INDEX}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera initialized!")
    print("\n" + "="*70)
    print("STARTING LIVE INFERENCE")
    print("="*70)
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'c' to toggle crop zoom")
    print("\nTips:")
    print("  - Center object in yellow detection box")
    print("  - Keep object close to camera")
    print("  - Use plain backgrounds for best results\n")
    
    # Main loop
    prev_time = time.time()
    fps = 0
    screenshot_count = 0
    use_crop = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Apply crop if enabled
            display_frame = crop_center(frame) if use_crop else frame
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Run prediction
            predicted_class, confidence = predict_fn(display_frame)
            
            # Draw overlay
            frame_with_overlay = draw_prediction_overlay(
                display_frame, predicted_class, confidence, fps, use_crop, model_name
            )
            
            # Display
            cv2.imshow(f'Office Classifier - {model_name.upper()}', frame_with_overlay)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{model_name}_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame_with_overlay)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('c'):
                use_crop = not use_crop
                mode = "ENABLED" if use_crop else "DISABLED"
                print(f"🔄 Crop zoom: {mode}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera released. Goodbye!")
        print("="*70)

if __name__ == '__main__':
    main()