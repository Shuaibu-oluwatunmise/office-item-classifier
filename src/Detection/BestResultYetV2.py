"""
Test Homography + PnP with Existing YOLO Model
Stabilized with RANSAC, Homography Inlier Check, and EMA Temporal Filter
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==============================================================================
# 🔴 CALIBRATION PARAMETERS
# ==============================================================================
CAMERA_MATRIX_CALIBRATED = np.array([
    [888.0265172956999, 0.0, 639.4063135490871],
    [0.0, 884.0984394681841, 354.6648324151391],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

DIST_COEFFS_FLAT = np.array([
    0.09450041549652369, 
    -0.22639662062971103, 
    0.0004947347614955076, 
    -0.0026696188964068354, 
    0.2021441235807068
], dtype=np.float32)
DIST_COEFFS_CALIBRATED = DIST_COEFFS_FLAT.reshape(5, 1)
# ==============================================================================


class NotebookPoseEstimator:
    def __init__(self, yolo_model_path, camera_index=1):
        """
        Initialize with existing YOLO model
        """
        self.camera_index = camera_index
        self.cap = None
        
        # Load YOLO model
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.class_names = self.yolo_model.names
        print(f"Model loaded with classes: {list(self.class_names.values())}")
        
        # Find notebook class ID
        self.notebook_class_id = next(
            (class_id for class_id, class_name in self.class_names.items() 
             if 'notebook' in class_name.lower()), None)
        
        if self.notebook_class_id is None:
            print("WARNING: 'notebook' class not found in model!")
        else:
            print(f"✓ Notebook class found: ID={self.notebook_class_id}")
            
        # Notebook physical dimensions (MEASURE YOUR NOTEBOOK!)
        self.notebook_width = 170.0    # mm (X dimension)
        self.notebook_height = 230.0   # mm (Y dimension)
        
        # Calculate half dimensions for centered origin
        half_w = self.notebook_width / 2.0
        half_h = self.notebook_height / 2.0
        
        # 3D object points (planar, Z=0) - Origin [0, 0, 0] is now the center.
        self.object_points_3d = np.array([
            [-half_w, half_h, 0],         # TL (Top-Left)
            [half_w, half_h, 0],          # TR (Top-Right)
            [half_w, -half_h, 0],         # BR (Bottom-Right)
            [-half_w, -half_h, 0]         # BL (Bottom-Left)
        ], dtype=np.float32)
        
        # Camera parameters (Hardcoded from calibration)
        self.camera_matrix = CAMERA_MATRIX_CALIBRATED
        self.dist_coeffs = DIST_COEFFS_CALIBRATED
        
        print("✓ Loaded CALIBRATED camera parameters for PnP.")
        
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Reference data
        self.ref_image = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_bbox_size = None
        
        # State
        self.calibrated = False
        
        # 🟢 STABILITY: Exponential Moving Average (EMA) Smoothing Variables
        self.rvec_smooth = None
        self.tvec_smooth = None
        # Alpha is the smoothing factor (0.2 is a good compromise for stability vs lag)
        self.smoothing_alpha = 0.2 
    
    def start_camera(self):
        """Initialize camera"""
        print(f"Starting camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_index}")
            return False
        
        # Setting the frame size to match the resolution used during calibration
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera started: {width}x{height}")
        return True
    
    def detect_notebook(self, frame):
        """
        Detect notebook using YOLO
        Returns bounding box [x1, y1, x2, y2] or None
        """
        results = self.yolo_model(frame, verbose=False)
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        # Find notebook detection with highest confidence
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
    
    def calibrate_reference(self, frame, bbox):
        """
        Calibrate reference from detected notebook (same logic, just captures ROI)
        """
        print("\n" + "="*60)
        print("CALIBRATING REFERENCE")
        print("="*60)
        
        x1, y1, x2, y2 = bbox
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Store reference
        self.ref_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.ref_bbox_size = (x2-x1, y2-y1)
        
        # Detect features
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.ref_image, None)
        
        if self.ref_descriptors is None or len(self.ref_keypoints) < 10:
            print("ERROR: Not enough features detected!")
            return False
        
        # Reset smooth filters on recalibration
        self.rvec_smooth = None
        self.tvec_smooth = None

        print(f"✓ Detected {len(self.ref_keypoints)} features")
        self.calibrated = True
        print("="*60)
        print("✓ CALIBRATION COMPLETE!")
        print("="*60 + "\n")
        
        return True
    
    def track_notebook_homography(self, frame, bbox):
        """
        Track notebook using feature matching and homography
        
        Returns:
            corners_2d: 4 corner points (TL, TR, BR, BL) or None
        """
        if not self.calibrated or self.ref_descriptors is None:
            return None
        
        # Get raw bbox and apply padding (Step 4)
        x1_raw, y1_raw, x2_raw, y2_raw = bbox

        # Pad the box a bit to include more context and reduce jitter
        pad = 10  # pixels
        h, w, _ = frame.shape
        x1 = max(0, x1_raw - pad)
        y1 = max(0, y1_raw - pad)
        x2 = min(w - 1, x2_raw + pad)
        y2 = min(h - 1, y2_raw + pad)

        roi = frame[y1:y2, x1:x2]
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)
        
        if descriptors is None or len(keypoints) < 10:
            return None
        
        # Match features
        matches = self.bf_matcher.knnMatch(self.ref_descriptors, descriptors, k=2)
        
        # Lowe's ratio test (Fixed)
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            return None
        
        # Get matched points
        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 🟢 STABILITY: Reject bad homographies (Step 2)
        if H is None or mask is None:
            return None

        # Calculate inlier ratio (sum of inliers / total matches used)
        inliers = mask.ravel().tolist()
        inlier_ratio = sum(inliers) / len(inliers)
        
        # Tune this threshold: 0.5 is default, 0.6 is stricter
        if inlier_ratio < 0.5:
            return None
        
        # Get reference corners (TL, TR, BR, BL order)
        h_ref, w_ref = self.ref_image.shape
        ref_corners = np.array([
            [0, 0],
            [w_ref, 0],
            [w_ref, h_ref],
            [0, h_ref]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform corners using homography
        corners_roi = cv2.perspectiveTransform(ref_corners, H)
        corners_roi = corners_roi.reshape(-1, 2)
        
        # Adjust corners to full frame coordinates (using the padded offset)
        corners_frame = corners_roi.copy()
        corners_frame[:, 0] += x1
        corners_frame[:, 1] += y1
        
        return corners_frame
    
    def estimate_pose_pnp(self, corners_2d):
        """
        Estimate 6DOF pose using PnP (Step 1: RANSAC for robustness)
        Returns raw rvec, tvec or (None, None) on failure
        """
        # corners_2d expected as (4, 2)
        image_points = corners_2d.reshape(-1, 1, 2).astype(np.float32)

        # Use solvePnPRansac for outlier rejection on 2D/3D points
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.object_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            reprojectionError=8.0, # Default threshold (tunes sensitivity)
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < 3:
            return None, None

        # Return raw vectors; smoothing is handled in run()
        return rvec, tvec
    
    def rotation_to_euler(self, rvec):
        """Convert rotation vector to Euler angles (degrees)"""
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Standard ZYX Euler extraction from rotation matrix
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        
        if sy > 1e-6:
            # Non-singularity case
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            # Singularity case (Gimbal lock)
            roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = 0
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    def draw_detection_box(self, frame, bbox):
        """Draw YOLO detection bounding box"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Notebook (YOLO)", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return frame
    
    def draw_tracked_corners(self, frame, corners):
        """Draw tracked corners from homography"""
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners_int], True, (0, 255, 0), 3)
        
        labels = ['TL', 'TR', 'BR', 'BL']
        for corner, label in zip(corners_int, labels):
            cv2.circle(frame, tuple(corner), 5, (0, 255, 0), -1)
            cv2.putText(frame, label, tuple(corner + [10, -10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def draw_3d_axes(self, frame, rvec, tvec, length=100):
        """
        Draw 3D coordinate axes from the calculated object origin (center).
        
        """
        # 3D points starting from the CENTER (0, 0, 0)
        axis_points_3d = np.float32([
            [0, 0, 0],           # Center of the object (Origin)
            [length, 0, 0],      # X-axis end
            [0, length, 0],      # Y-axis end
            [0, 0, length]       # Z-axis end
        ])
        
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        origin = tuple(axis_points_2d[0])
        
        # Draw axes
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3, tipLength=0.3)  # X-Red 
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3, tipLength=0.3)  # Y-Green 
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3, tipLength=0.3)  # Z-Blue 
        
        # Labels
        cv2.putText(frame, 'X', tuple(axis_points_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, 'Y', tuple(axis_points_2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, 'Z', tuple(axis_points_2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("YOLO DETECTION + HOMOGRAPHY + PnP TEST")
        print("STABILITY: RANSAC, Inlier Check, and EMA Filter Applied")
        print("="*60)
        print("\nControls:")
        print("  'c' - Calibrate (capture reference)")
        print("  'r' - Reset calibration")
        print("  's' - Save frame")
        print("  ESC - Exit")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return
        
        frame_count = 0
        saved_count = 0
        fps = 0
        last_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # Detect notebook with YOLO
                bbox = self.detect_notebook(frame)
                
                if bbox is not None:
                    # Draw detection box
                    display_frame = self.draw_detection_box(display_frame, bbox)
                    
                    if not self.calibrated:
                        cv2.putText(display_frame, "Press 'c' to calibrate", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        # Track with homography
                        corners = self.track_notebook_homography(frame, bbox)
                        
                        if corners is not None:
                            # Estimate pose (Raw)
                            rvec_raw, tvec_raw = self.estimate_pose_pnp(corners)
                            
                            if rvec_raw is not None and tvec_raw is not None:
                                
                                # 🟢 STABILITY: Pose smoothing (Step 3)
                                if self.rvec_smooth is None:
                                    self.rvec_smooth = rvec_raw.copy()
                                    self.tvec_smooth = tvec_raw.copy()
                                else:
                                    alpha = self.smoothing_alpha
                                    # Use the raw PnP result to update the smoothed result
                                    self.rvec_smooth = (1 - alpha) * self.rvec_smooth + alpha * rvec_raw
                                    self.tvec_smooth = (1 - alpha) * self.tvec_smooth + alpha * tvec_raw
                                
                                # Use smoothed vectors for output
                                rvec_final = self.rvec_smooth
                                tvec_final = self.tvec_smooth

                                # Draw tracked corners
                                display_frame = self.draw_tracked_corners(display_frame, corners)
                                
                                # Draw 3D axes
                                display_frame = self.draw_3d_axes(display_frame, rvec_final, tvec_final)
                                
                                # Get Euler angles
                                roll, pitch, yaw = self.rotation_to_euler(rvec_final)
                                
                                # tvec gives the [X, Y, Z] position of the CENTER in the camera's frame
                                distance_z = tvec_final[2, 0]
                                
                                # Display pose info
                                y_offset = 30
                                cv2.putText(display_frame, f"Z Distance: {distance_z:.0f}mm", (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                y_offset += 30
                                cv2.putText(display_frame, f"Roll (X): {roll:.1f}deg", (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
                                y_offset += 30
                                cv2.putText(display_frame, f"Pitch (Y): {pitch:.1f}deg", (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
                                y_offset += 30
                                cv2.putText(display_frame, f"Yaw (Z): {yaw:.1f}deg", (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                
                                cv2.putText(display_frame, "TRACKING OK (Stable)", (10, display_frame.shape[0]-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.putText(display_frame, "Pose RANSAC failed", (10, display_frame.shape[0]-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        else:
                            cv2.putText(display_frame, "Homography/Inlier failed", (10, display_frame.shape[0]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "No notebook detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show FPS
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Notebook Pose Estimation Test', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                elif key == ord('c') and bbox is not None:  # Calibrate
                    if self.calibrate_reference(frame, bbox):
                        print("✓ Ready for tracking!")
                elif key == ord('r'):  # Reset
                    self.calibrated = False
                    self.ref_image = None
                    self.ref_keypoints = None
                    self.ref_descriptors = None
                    self.rvec_smooth = None  # Reset smoother
                    self.tvec_smooth = None  # Reset smoother
                    print("✓ Calibration reset")
                elif key == ord('s'):  # Save
                    filename = f"test_result_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    print(f"✓ Saved: {filename}")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("SETUP")
    print("="*60)
    print("✓ Using hardcoded camera calibration (Reprojection Error: 0.1972).")
    print("✓ 3D object points are centered at [0, 0, 0].")
    
    # UPDATE THIS PATH TO YOUR TRAINED MODEL
    yolo_model_path = '../../runs/detect/yolov8n_detect_V5/weights/best.pt'
    
    camera_index = 1
    
    # Instantiate the estimator object first
    estimator = NotebookPoseEstimator(yolo_model_path, camera_index)
    
    # Accessing attributes through the 'estimator' instance
    print(f"1. Check if notebook size ({estimator.notebook_width}x{estimator.notebook_height}mm) is correct.")
    print(f"2. Pose Smoothing Alpha: {estimator.smoothing_alpha} (0.2 is a low smoothing value)")
    print("="*60 + "\n")

    estimator.run()


if __name__ == "__main__":
    main()