#Detection/File_inference.py
from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np

def test_yolov8_detection():
    # Configuration
    model_path = 'runs/detect/yolov8n_detect_V5/weights/best.pt'  # Path to your trained model
    input_folder = 'my_data'                      # Folder containing images to test
    output_folder = 'detection_results'               # Folder to save annotated images
    confidence_threshold = 0.25                       # Minimum confidence to show detection
    iou_threshold = 0.45                              # Non-maximum suppression threshold
    
    # Text display settings - INCREASED SIZE
    font_scale = 1.8                                  # Much larger font (was 0.6)
    font_thickness = 4                                # Thicker text (was 2)
    box_thickness = 3                                 # Thicker bounding boxes
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load your trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get class names from the model
    class_names = model.names
    print(f"Model loaded with {len(class_names)} classes: {list(class_names.values())}")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    print(f"Found {len(image_paths)} images in '{input_folder}'")
    
    if not image_paths:
        print("No images found! Please check your input folder path.")
        return
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Could not read image: {image_path}")
            continue
        
        # Run inference
        results = model(image, conf=confidence_threshold, iou=iou_threshold, verbose=False)
        
        # Extract detections
        result = results[0]
        boxes = result.boxes
        
        # Draw detections on image
        annotated_image = image.copy()
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = class_names[class_id]
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Choose color based on class (you can customize this)
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                         (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
                color = colors[class_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)
                
                # Create label text
                label = f"{class_name}: {confidence:.2f}"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw background for text (with extra padding)
                text_bg_x1 = x1
                text_bg_y1 = max(0, y1 - text_height - baseline - 10)  # Ensure it doesn't go above image
                text_bg_x2 = x1 + text_width + 10
                text_bg_y2 = y1
                
                cv2.rectangle(
                    annotated_image, 
                    (text_bg_x1, text_bg_y1), 
                    (text_bg_x2, text_bg_y2), 
                    color, 
                    -1  # Filled rectangle
                )
                
                # Draw text (white with black outline for maximum readability)
                text_position = (x1 + 5, y1 - baseline - 5)
                
                # First draw black outline
                cv2.putText(
                    annotated_image, 
                    label, 
                    text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 0, 0),  # Black outline
                    font_thickness + 1
                )
                
                # Then draw white text
                cv2.putText(
                    annotated_image, 
                    label, 
                    text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 255, 255),  # White text
                    font_thickness
                )
            
            print(f"  Detected {len(boxes)} objects")
        else:
            print("  No detections")
        
        # Save annotated image
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_image)
    
    print(f"\nProcessing complete! Annotated images saved to '{output_folder}'")

def test_single_image_with_details():
    """Alternative function to test a single image and print detailed results"""
    model_path = 'runs/detect/train/weights/best.pt'
    image_path = 'test_image.jpg'  # Change to your test image
    
    # Text settings for single image (can be even larger)
    font_scale = 2.0
    font_thickness = 5
    box_thickness = 4
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)
    result = results[0]
    
    # Print detailed results
    print("\n" + "="*50)
    print("DETAILED DETECTION RESULTS")
    print("="*50)
    
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"Detected {len(result.boxes)} objects:")
        print("-" * 50)
        
        # Get the original image
        image = cv2.imread(image_path)
        annotated_image = image.copy()
        
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            print(f"Object {i+1}:")
            print(f"  Class: {class_name} (ID: {class_id})")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            print(f"  Width: {x2-x1:.1f}, Height: {y2-y1:.1f}")
            print("-" * 30)
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Choose color
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw background
            cv2.rectangle(
                annotated_image, 
                (x1, max(0, y1 - text_height - baseline - 10)), 
                (x1 + text_width + 10, y1), 
                color, 
                -1
            )
            
            # Draw text with outline
            text_pos = (x1 + 5, y1 - baseline - 5)
            
            # Black outline
            cv2.putText(
                annotated_image, 
                label, 
                text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 0, 0), 
                font_thickness + 2
            )
            
            # White text
            cv2.putText(
                annotated_image, 
                label, 
                text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                font_thickness
            )
        
        # Display the image
        cv2.imshow('Detection Results - Press any key to close', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the annotated image
        cv2.imwrite('single_image_result.jpg', annotated_image)
        print("Annotated image saved as 'single_image_result.jpg'")
    else:
        print("No objects detected!")

if __name__ == "__main__":
    # Run batch processing on folder of images
    test_yolov8_detection()
    
    # Uncomment the line below to test a single image with even larger text
    # test_single_image_with_details()