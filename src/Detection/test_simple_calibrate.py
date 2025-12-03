"""
Press 'C' to Calibrate - Simple & Responsive
YOLO detects → Press C → Track with optical flow
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time


class SimpleCalibrateTracker:
    def __init__(self, yolo_model_path, camera_index=1):
        self.camera_index = camera_index
        self.cap = None
        
        # Load YOLO
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.class_names = self.yolo_model.names
        
        # Find notebook class
        self.notebook_class_id = None
        for class_id, class_name in self.class_names.items():
            if 'notebook' in class_name.lower():
                self.notebook_class_id = class_id
                print(f"✓ Notebook class: '{class_name}'")
                break
        
        # Object dimensions
        self.notebook_width = 170.0
        self.notebook_height = 230.0
        
        # 3D points - CENTERED
        hw = self.notebook_width / 2
        hh = self.notebook_height / 2
        self.object_points_3d = np.array([
            [-hw, -hh, 0],  # TL
            [ hw, -hh, 0],  # TR
            [ hw,  hh, 0],  # BR
            [-hw,  hh, 0]   # BL
        ], dtype=np.float32)
        
        # Camera
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Tracking
        self.tracked_corners = None
        self.prev_gray = None
        self.calibrated = False
        
        # Optical flow params - TUNED FOR SPEED
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # MINIMAL smoothing for responsiveness
        self.prev_rvec = None
        self.prev_tvec = None
        self.alpha = 0.4  # 40% new, 60% old
        
    def load_camera_calibration(self):
        try:
            import sys, os
            calib_path = os.path.join(os.path.dirname(__file__), 'camera_calibration')
            if calib_path not in sys.path:
                sys.path.insert(0, calib_path)
            from camera_params import CAMERA_MATRIX, DIST_COEFFS
            self.camera_matrix = CAMERA_MATRIX
            self.dist_coeffs = DIST_COEFFS.reshape(-1, 1)
            print("✓ Loaded calibration")
            return True
        except:
            return False
    
    def estimate_camera_matrix(self, width, height):
        focal_length = width * 1.2
        center = (width / 2, height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    def start_camera(self):
        print(f"Starting camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not self.load_camera_calibration():
            self.estimate_camera_matrix(width, height)
        
        print(f"✓ Camera: {width}x{height}")
        return True
    
    def detect_notebook(self, frame):
        results = self.yolo_model(frame, verbose=False)
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        best_bbox = None
        best_conf = 0
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id == self.notebook_class_id and confidence > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_bbox = [x1, y1, x2, y2]
                best_conf = confidence
        
        return best_bbox
    
    def calibrate_from_bbox(self, bbox):
        """Capture 4 corners from YOLO bbox"""
        x1, y1, x2, y2 = bbox
        
        self.tracked_corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        
        self.calibrated = True
        self.prev_rvec = None
        self.prev_tvec = None
        
        print("="*60)
        print("✓ CALIBRATED! Tracking 4 corners with optical flow")
        print("="*60)
    
    def track_corners(self, gray):
        """Track 4 corners with optical flow"""
        if self.tracked_corners is None or self.prev_gray is None:
            return None
        
        # Track with optical flow
        new_corners, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.tracked_corners.reshape(-1, 1, 2),
            None,
            **self.lk_params
        )
        
        if new_corners is None or not np.all(status == 1):
            return None
        
        self.tracked_corners = new_corners.reshape(-1, 2)
        return self.tracked_corners
    
    def estimate_pose(self, corners):
        """Estimate pose with LIGHT smoothing"""
        success, rvec, tvec = cv2.solvePnP(
            self.object_points_3d,
            corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None
        
        # Light exponential smoothing
        if self.prev_rvec is not None:
            rvec = self.alpha * rvec + (1 - self.alpha) * self.prev_rvec
            tvec = self.alpha * tvec + (1 - self.alpha) * self.prev_tvec
        
        self.prev_rvec = rvec.copy()
        self.prev_tvec = tvec.copy()
        
        return rvec, tvec
    
    def rotation_to_euler(self, rvec):
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        
        if sy > 1e-6:
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = 0
        
        return (np.degrees(roll), np.degrees(pitch), np.degrees(yaw))
    
    def draw_bbox(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, "DETECTED", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return frame
    
    def draw_corners(self, frame, corners):
        corners_int = corners.astype(int)
        
        # Green box
        cv2.polylines(frame, [corners_int], True, (0, 255, 0), 4, cv2.LINE_AA)
        
        # Corner circles
        for i, corner in enumerate(corners_int):
            cv2.circle(frame, tuple(corner), 8, (0, 255, 0), -1)
            cv2.circle(frame, tuple(corner), 10, (255, 255, 255), 2)
        
        return frame
    
    def draw_axes(self, frame, rvec, tvec, length=120):
        """Draw axes at CENTER"""
        axis_3d = np.float32([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length]
        ])
        
        axis_2d, _ = cv2.projectPoints(
            axis_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        axis_2d = axis_2d.reshape(-1, 2).astype(int)
        origin = tuple(axis_2d[0])
        
        # THICK axes
        cv2.arrowedLine(frame, origin, tuple(axis_2d[1]), (0, 0, 255), 7, 
                       cv2.LINE_AA, tipLength=0.3)  # X-Red
        cv2.arrowedLine(frame, origin, tuple(axis_2d[2]), (0, 255, 0), 7, 
                       cv2.LINE_AA, tipLength=0.3)  # Y-Green
        cv2.arrowedLine(frame, origin, tuple(axis_2d[3]), (255, 0, 0), 7, 
                       cv2.LINE_AA, tipLength=0.3)  # Z-Blue
        
        # Labels
        for pt, label, color in [(axis_2d[1], 'X', (0,0,255)),
                                  (axis_2d[2], 'Y', (0,255,0)),
                                  (axis_2d[3], 'Z', (255,0,0))]:
            pos = tuple(pt)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        return frame
    
    def run(self):
        print("\n" + "="*60)
        print("SIMPLE CALIBRATE TRACKER")
        print("="*60)
        print("\nHow to use:")
        print("  1. Point camera at notebook")
        print("  2. Wait for BLUE detection box")
        print("  3. Press 'c' to CALIBRATE")
        print("  4. Tracking starts!")
        print("\nControls:")
        print("  'c' - Calibrate (when detected)")
        print("  'r' - Reset")
        print("  's' - Save")
        print("  ESC - Exit")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return
        
        frame_count = 0
        fps = 0
        last_time = time.time()
        last_bbox = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = frame.copy()
                
                # FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                if not self.calibrated:
                    # Detection mode
                    bbox = self.detect_notebook(frame)
                    
                    if bbox is not None:
                        last_bbox = bbox
                        display = self.draw_bbox(display, bbox)
                        
                        cv2.putText(display, "Press 'C' to CALIBRATE", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1.5, (0, 255, 255), 3)
                    else:
                        cv2.putText(display, "Point at notebook...", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1.3, (255, 255, 0), 3)
                
                else:
                    # Tracking mode
                    corners = self.track_corners(gray)
                    
                    if corners is not None:
                        display = self.draw_corners(display, corners)
                        
                        rvec, tvec = self.estimate_pose(corners)
                        
                        if rvec is not None:
                            display = self.draw_axes(display, rvec, tvec)
                            
                            roll, pitch, yaw = self.rotation_to_euler(rvec)
                            distance = np.linalg.norm(tvec)
                            
                            # Info
                            y = 45
                            cv2.putText(display, f"Dist: {distance:.0f}mm", 
                                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                            y += 35
                            cv2.putText(display, f"Roll:  {roll:.1f}", 
                                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            y += 30
                            cv2.putText(display, f"Pitch: {pitch:.1f}", 
                                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                            y += 30
                            cv2.putText(display, f"Yaw:   {yaw:.1f}", 
                                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                            
                            cv2.putText(display, "TRACKING", 
                                       (10, display.shape[0]-15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                    else:
                        cv2.putText(display, "LOST - Press 'r' to reset", 
                                   (10, display.shape[0]-15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                
                # FPS
                cv2.putText(display, f"FPS: {fps}", 
                           (display.shape[1]-150, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                
                self.prev_gray = gray
                cv2.imshow('Simple Calibrate Tracker', display)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('c'):
                    if not self.calibrated and last_bbox is not None:
                        self.calibrate_from_bbox(last_bbox)
                    elif self.calibrated:
                        print("Already calibrated! Press 'r' to reset first")
                elif key == ord('r'):
                    self.calibrated = False
                    self.tracked_corners = None
                    self.prev_rvec = None
                    self.prev_tvec = None
                    print("✓ Reset - press 'c' to calibrate again")
                elif key == ord('s'):
                    cv2.imwrite(f"result_{int(time.time())}.jpg", display)
                    print("✓ Saved")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    yolo_model_path = '../../runs/detect/yolov8n_detect_V5/weights/best.pt'
    tracker = SimpleCalibrateTracker(yolo_model_path, camera_index=1)
    tracker.run()


if __name__ == "__main__":
    main()