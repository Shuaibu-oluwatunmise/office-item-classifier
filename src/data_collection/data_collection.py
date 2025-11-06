import cv2
import os
from datetime import datetime
import time

def create_directory(class_name):
    """Create directory for the class if it doesn't exist"""
    dir_path = os.path.join("Data_Video", class_name.lower())
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_video_filename(dir_path):
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{timestamp}.avi"
    return os.path.join(dir_path, filename)

def record_video(class_name, duration=20):
    """Record video for specified duration"""
    # Create directory
    dir_path = create_directory(class_name)
    video_path = get_video_filename(dir_path)
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default FPS if camera doesn't report it
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"\nRecording for {duration} seconds...")
    print("Press 'q' to stop recording early")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Write frame to video file
        out.write(frame)
        
        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = duration - elapsed_time
        
        # Display frame with timer
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Recording: {remaining_time:.1f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Class: {class_name}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recording', display_frame)
        
        # Check if duration reached or 'q' pressed
        if elapsed_time >= duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved to: {video_path}")
    return True

def main():
    """Main function to handle data collection"""
    print("=" * 50)
    print("Object Detection Data Collection Tool")
    print("=" * 50)
    print("\nType 'quit' or 'exit' to stop")
    print("=" * 50)
    
    while True:
        # Get class name from user
        class_input = input("\nEnter the class name: ").strip()
        
        # Check for exit command
        if class_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting data collection tool. Goodbye!")
            break
        
        # Check for empty input
        if not class_input:
            print("Error: Class name cannot be empty. Please try again.")
            continue
        
        # Convert to lowercase for processing
        class_name = class_input.lower()
        
        # Record video
        success = record_video(class_name)
        
        if not success:
            print("Recording failed. Please try again.")
        else:
            print(f"✓ Successfully recorded video for class: {class_name}")

if __name__ == "__main__":
    main()