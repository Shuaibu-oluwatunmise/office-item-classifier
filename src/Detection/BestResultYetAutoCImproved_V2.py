"""
Test Homography + PnP with Existing YOLO Model
Multi-object: Notebook + Keyboard
Stabilized with Per-Object Thresholds (RANSAC, Homography Inlier Check, and EMA Temporal Filter)

*** MODIFIED for ROBUST "BEST-SHOT" AUTOCALIBRATION ***
- detect_objects now returns YOLO confidences.
- Added should_auto_calibrate gate to ensure we only consider candidate frames
  on high-confidence, reasonably sized detections.                
- Added BEST-OF-5 mechanism: When gated candidates appear, buffer 5 frames 
  and select the one with the maximum ORB keypoints as the final reference.
- The 'c' key is no longer functional.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==============================================================================
# 1. OBJECT CONFIGS (NOW WITH PER-OBJECT TRACKING TUNING)
# ==============================================================================
OBJECT_CONFIGS = {
    "notebook": {
        "label_substring": "notebook",   # substring to match YOLO class name
        "width_mm": 170.0,               # your measured notebook width
        "height_mm": 230.0,              # your measured notebook height
        "axis_color": (0, 0, 255),       # Color for Notebook info (Red)
        # tracking tuning (Notebook: Snappy & Strict)
        "min_matches": 10,
        "min_inlier_ratio": 0.5,
        "ema_alpha": 0.2,
    },
    "keyboard": {
        "label_substring": "keyboard",   # substring to match YOLO class name
        "width_mm": 312.0,               # measured keyboard width
        "height_mm": 146.0,              # measured keyboard height
        "axis_color": (255, 255, 0),     # Color for Keyboard info (Cyan/Yellow)
        # tracking tuning (Keyboard: Relaxed & Smoothed)
        "min_matches": 6,                # Reduced minimum matches
        "min_inlier_ratio": 0.3,         # Reduced minimum inlier ratio
        "ema_alpha": 0.1,                # Increased smoothing (lower alpha)
    },
}
# ==============================================================================

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


class MultiObjectPoseEstimator:
    def __init__(self, yolo_model_path, camera_index=1):
        self.camera_index = camera_index
        self.cap = None

        # Load YOLO model
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.class_names = self.yolo_model.names
        print(f"Model loaded with classes: {list(self.class_names.values())}")

        # ------------------------------------------------------------------
        # Build per-object state (class id, size, 3D points, tracking state)
        # ------------------------------------------------------------------
        self.targets = {}   # name -> state dict

        for obj_name, cfg in OBJECT_CONFIGS.items():
            label_sub = cfg["label_substring"].lower()
            class_id = next(
                (
                    cid for cid, cname in self.class_names.items()
                    if label_sub in cname.lower()
                ),
                None,
            )
            if class_id is None:
                print(f"WARNING: '{obj_name}' class not found in model (substring '{label_sub}')!")
            else:
                print(f"✓ {obj_name.capitalize()} class found: ID={class_id}")

            width_mm = cfg["width_mm"]
            height_mm = cfg["height_mm"]
            half_w = width_mm / 2.0
            half_h = height_mm / 2.0

            # 3D object points (planar, Z=0) - Origin [0, 0, 0] is the center.
            object_points_3d = np.array([
                [-half_w,  half_h, 0],   # TL (Top-Left)
                [ half_w,  half_h, 0],   # TR (Top-Right)
                [ half_w, -half_h, 0],   # BR (Bottom-Right)
                [-half_w, -half_h, 0],   # BL (Bottom-Left)
            ], dtype=np.float32)

            self.targets[obj_name] = {
                "class_id": class_id,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "object_points_3d": object_points_3d,
                "axis_color": cfg["axis_color"],
                
                # Per-object thresholds
                "min_matches": cfg.get("min_matches", 10),
                "min_inlier_ratio": cfg.get("min_inlier_ratio", 0.5),
                "ema_alpha": cfg.get("ema_alpha", 0.2),

                # Reference ORB data
                "ref_image": None,
                "ref_keypoints": None,
                "ref_descriptors": None,
                "ref_bbox_size": None,

                # State
                "calibrated": False,
                "rvec_smooth": None,
                "tvec_smooth": None,

                # 🟢 NEW: Multi-frame "Best Shot" buffer
                "calib_buffer": [],
                "calib_buffer_size": 5,  # Buffer 5 "good" candidate frames before locking
            }

        # Camera parameters
        self.camera_matrix = CAMERA_MATRIX_CALIBRATED
        self.dist_coeffs = DIST_COEFFS_CALIBRATED

        print("✓ Loaded CALIBRATED camera parameters for PnP.")

        # ORB with more features
        self.orb = cv2.ORB_create(nfeatures=4000) # Bumped from 2000
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        print(f"✓ ORB initialized with nfeatures=4000.")


    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def start_camera(self):
        print(f"Starting camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera started: {width}x{height}")
        return True

    # ------------------------------------------------------------------
    # Detection: return bboxes + confidences
    # ------------------------------------------------------------------
    def detect_objects(self, frame):
        """
        Returns:
            bboxes:      dict name -> bbox [x1,y1,x2,y2] or None
            confidences: dict name -> confidence (0..1)
        """
        results = self.yolo_model(frame, verbose=False)
        result = results[0]

        bboxes = {name: None for name in self.targets}
        best_conf = {name: 0.0 for name in self.targets}

        if result.boxes is None or len(result.boxes) == 0:
            return bboxes, best_conf

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            for name, state in self.targets.items():
                if state["class_id"] is None:
                    continue
                if class_id == state["class_id"] and confidence > best_conf[name]:
                    bboxes[name] = [x1, y1, x2, y2]
                    best_conf[name] = confidence

        return bboxes, best_conf

    # ------------------------------------------------------------------
    # Calibration per object (single ROI)
    # ------------------------------------------------------------------
    def calibrate_reference(self, frame, obj_name, bbox):
        """
        Calibrate reference for a specific object from its detected bbox.                
        NOTE: 'frame' can be the full frame or a pre-cropped candidate ROI.
        """
        state = self.targets[obj_name]
        print("\n" + "="*60)
        print(f"CALIBRATING FROM BEST CANDIDATE: {obj_name.upper()}")
        print("="*60)

        x1, y1, x2, y2 = bbox
        
        # Determine if 'frame' is the full frame or already cropped
        if frame.shape[0] == state["ref_bbox_size"]: # Rough check
             roi = frame
        else:
             roi = frame[y1:y2, x1:x2]

        state["ref_image"] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        state["ref_bbox_size"] = (x2 - x1, y2 - y1)
        state["ref_keypoints"], state["ref_descriptors"] = self.orb.detectAndCompute(
            state["ref_image"], None
        )

        if state["ref_descriptors"] is None or len(state["ref_keypoints"]) < 10:
            print("ERROR: Not enough features in this candidate! Retrying next window.")                
            state["calibrated"] = False
            return False

        # Reset smoothing
        state["rvec_smooth"] = None
        state["tvec_smooth"] = None

        print(f"✓ Detected {len(state['ref_keypoints'])} features")
        print(f"   Using min_matches={state['min_matches']}, min_inlier_ratio={state['min_inlier_ratio']}, ema_alpha={state['ema_alpha']}")
        state["calibrated"] = True
        print("="*60)
        print(f"✓ CALIBRATION COMPLETE for {obj_name.upper()}!")
        print("="*60 + "\n")
        return True

    # ------------------------------------------------------------------
    # Gate Heuristic for Autocalibration
    # ------------------------------------------------------------------
    def should_auto_calibrate(self, obj_name, bbox, confidence, frame_shape):
        """
        Decide whether it's a good moment to CONSIDER auto-calibrating this object.                
        Heuristics:
          - YOLO confidence must be reasonably high
          - Bbox size must not be tiny (object too far away)
        """
        if confidence < 0.7:                # YOLO confidence threshold
            return False

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        H, W = frame_shape[:2]
        area = w * h
        frame_area = W * H

        # Require object to occupy at least ~2% of the frame
        if area < 0.02 * frame_area:
            return False

        # Per-object tweak: keyboard needs to be a bit larger/more central
        if obj_name == "keyboard":
            if area < 0.03 * frame_area:
                return False

        return True

    # ------------------------------------------------------------------
    # 🟢 5. NEW: Buffer Candidates for "Best Shot"
    # ------------------------------------------------------------------
    def accumulate_and_maybe_calibrate(self, frame, obj_name, bbox):
        """
        Collects candidate ROIs over multiple frames and calibrates once
        we have enough, using the ROI with the most keypoints.
        Returns True iff calibration was performed successfully.
        """
        state = self.targets[obj_name]
        x1, y1, x2, y2 = bbox
        
        # Crop the candidate ROI immediately to save memory
        roi = frame[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            # Not a useful candidate; skip without adding to buffer
            return False

        state["calib_buffer"].append({
            "roi": roi,
            "n_kp": len(keypoints),
        })

        # Not enough candidates to pick the best yet
        if len(state["calib_buffer"]) < state["calib_buffer_size"]:
            print(f"Buffered candidate for {obj_name} ({len(state['calib_buffer'])}/{state['calib_buffer_size']})")                
            return False

        # --- BUFFER FULL: Perform Best-Frame Selection ---
        print(f"Buffer full for {obj_name}. Selecting best frame...")
        
        # Pick the candidate with the absolute maximum number of keypoints
        best_candidate = max(state["calib_buffer"], key=lambda c: c["n_kp"])
        roi_best = best_candidate["roi"]

        # Run the standard calibration on this best ROI
        # Prepare dummy bbox that covers full ROI area
        h_best, w_best = roi_best.shape[:2]
        bbox_best = [0, 0, w_best, h_best]
        success = self.calibrate_reference(roi_best, obj_name, bbox_best)

        # Clear buffer for next time (or retry if failed)
        state["calib_buffer"].clear()

        return success

    # ------------------------------------------------------------------
    # Homography tracking per object
    # ------------------------------------------------------------------
    def track_object_homography(self, frame, obj_name, bbox):
        state = self.targets[obj_name]

        if not state["calibrated"] or state["ref_descriptors"] is None:
            return None

        # Get per-object thresholds
        min_matches = state["min_matches"]
        min_inlier_ratio = state["min_inlier_ratio"]

        x1_raw, y1_raw, x2_raw, y2_raw = bbox

        # Pad the box a bit to include more context and reduce jitter
        pad = 10
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

        matches = self.bf_matcher.knnMatch(state["ref_descriptors"], descriptors, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < min_matches:
            return None

        src_pts = np.float32(
            [state["ref_keypoints"][m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None

        # Check inlier ratio
        inliers = mask.ravel().tolist()
        inlier_ratio = sum(inliers) / len(inliers)
        if inlier_ratio < min_inlier_ratio:
            return None

        # Transform reference corners to the current frame
        h_ref, w_ref = state["ref_image"].shape
        ref_corners = np.array(
            [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32
        ).reshape(-1, 1, 2)

        corners_roi = cv2.perspectiveTransform(ref_corners, H)
        corners_roi = corners_roi.reshape(-1, 2)

        # Adjust corners to full frame coordinates (using the padded offset)
        corners_frame = corners_roi.copy()
        corners_frame[:, 0] += x1
        corners_frame[:, 1] += y1

        return corners_frame

    # ------------------------------------------------------------------
    # Pose Estimation (PnP)
    # ------------------------------------------------------------------
    def estimate_pose_pnp(self, obj_name, corners_2d):
        """
        Estimate 6DOF pose using PnP (RANSAC for robustness)
        Returns raw rvec, tvec or (None, None) on failure
        """
        state = self.targets[obj_name]
        
        # corners_2d expected as (4, 2)
        image_points = corners_2d.reshape(-1, 1, 2).astype(np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            state["object_points_3d"], 
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            reprojectionError=8.0, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < 3:
            return None, None

        return rvec, tvec
    
    # ------------------------------------------------------------------
    # Pose Utility
    # ------------------------------------------------------------------
    def rotation_to_euler(self, rvec):
        """Convert rotation vector to Euler angles (degrees) using ZYX convention"""
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
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    # ------------------------------------------------------------------
    # Drawing Utilities
    # ------------------------------------------------------------------
    def draw_detection_box(self, frame, obj_name, bbox):
        """Draw YOLO detection bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Color based on calibration state
        state = self.targets[obj_name]
        color = (0, 255, 0) if state["calibrated"] else (255, 0, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{obj_name.capitalize()} (YOLO)", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
    
    def draw_tracked_corners(self, frame, obj_name, corners):
        """Draw tracked corners from homography"""
        corners_int = corners.astype(int)
        color = self.targets[obj_name]["axis_color"] 
        cv2.polylines(frame, [corners_int], True, color, 3)
        return frame
    
    def draw_3d_axes(self, frame, obj_name, rvec, tvec, length=100):
        """Draw 3D coordinate axes from the calculated object origin (center)."""
        color = self.targets[obj_name]["axis_color"]
        
        axis_points_3d = np.float32([
            [0, 0, 0],       # Center of the object (Origin)
            [length, 0, 0],  # X-axis end
            [0, length, 0],  # Y-axis end
            [0, 0, length]   # Z-axis end
        ])
        
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        origin = tuple(axis_points_2d[0])
        
        # Draw axes (X=Red, Y=Green, Z=Blue) 
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3, tipLength=0.3)  # X-Red 
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3, tipLength=0.3)  # Y-Green 
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3, tipLength=0.3)  # Z-Blue 
        
        cv2.circle(frame, origin, 8, color, -1)
        
        return frame

    def draw_pose_info(self, frame, obj_name, y_offset_start, rvec_final, tvec_final):
        """Draws RPY and Distance info for a single object at a specific y_offset."""
        
        roll, pitch, yaw = self.rotation_to_euler(rvec_final)
        distance_z = tvec_final[2, 0] # Z distance in mm
        
        text_color = self.targets[obj_name]["axis_color"]
        y_offset = y_offset_start
        
        # Header
        cv2.putText(frame, f"--- {obj_name.upper()} ---", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        y_offset += 25
        
        # Translations
        cv2.putText(frame, f"X,Y,Z Pos: [{tvec_final[0, 0]:.0f}, {tvec_final[1, 0]:.0f}, {distance_z:.0f}]mm", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Rotations
        cv2.putText(frame, f"Roll (X): {roll:.1f}deg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Red X
        y_offset += 25
        cv2.putText(frame, f"Pitch (Y): {pitch:.1f}deg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Green Y
        y_offset += 25
        cv2.putText(frame, f"Yaw (Z): {yaw:.1f}deg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Blue Z
        y_offset += 25
        
        return y_offset

    # ------------------------------------------------------------------
    # Main Loop (Gated + Multi-frame Best-Shot Autocalibration)
    # ------------------------------------------------------------------
    def run(self):
        """Main loop with automatic, opportunistic multi-frame calibration."""
        print("\n" + "="*60)
        print("YOLO DETECTION + HOMOGRAPHY + PnP TEST (MULTI-OBJECT)")
        print("STABILITY: RANSAC, Per-Object Thresholds, Per-Object EMA")
        print("*** AUTOCALIBRATION: Gated + Multi-frame Best Shot ***")                
        print("="*60)
        print("\nControls:")
        print("  'r' - Reset ALL calibrations (Will recalibrate)")
        print("  's' - Save frame")
        print("  ESC - Exit")
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
                
                # DETECTION: Unpack bboxes and confidences
                bboxes, confidences = self.detect_objects(frame)
                
                # Info display offset
                y_info_offset = 30
                tracked_objects_count = 0
                
                detected_objects = [
                    obj_name for obj_name, bbox in bboxes.items() if bbox is not None
                ]
                
                if len(detected_objects) == 0:
                     cv2.putText(display_frame, "No object detected", (10, display_frame.shape[0]-10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                
                for obj_name in detected_objects:
                    bbox = bboxes[obj_name]
                    state = self.targets[obj_name]
                    
                    # Draw YOLO detection box
                    display_frame = self.draw_detection_box(display_frame, obj_name, bbox)
                    
                    # 🟢 AUTOCALIBRATION LOGIC (Gated + Multi-frame)
                    if not state["calibrated"]:
                        conf = confidences.get(obj_name, 0.0)
                        
                        # Step 1: Check sanity gate (Size/Conf)
                        if self.should_auto_calibrate(obj_name, bbox, conf, frame.shape):
                            
                            # Step 2: Buffer and maybe calibrate from best shot
                            if not self.accumulate_and_maybe_calibrate(frame, obj_name, bbox):
                                # Still filling buffer or just failed -> skip tracking
                                continue                
                        else:
                            # Not a good frame -> wait
                            continue
                    
                    # Tracking proceeds only if calibrated
                    if state["calibrated"]: 
                        
                        # Track with homography
                        corners = self.track_object_homography(frame, obj_name, bbox)
                        
                        if corners is not None:
                            # Estimate pose (Raw)
                            rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                            
                            if rvec_raw is not None and tvec_raw is not None:
                                
                                # EMA Temporal Smoothing
                                alpha = state.get("ema_alpha", 0.2) 
                                if state["rvec_smooth"] is None:
                                    state["rvec_smooth"] = rvec_raw.copy()
                                    state["tvec_smooth"] = tvec_raw.copy()
                                else:
                                    state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                                    state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                                
                                rvec_final = state["rvec_smooth"]
                                tvec_final = state["tvec_smooth"]

                                # Draw tracked corners and 3D axes
                                display_frame = self.draw_tracked_corners(display_frame, obj_name, corners)
                                display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final)
                                
                                # Display pose info
                                y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, rvec_final, tvec_final)
                                tracked_objects_count += 1
                                
                            else:
                                # PnP Failed
                                y_info_offset += 25
                                cv2.putText(display_frame, f"Pose RANSAC failed for {obj_name}", (10, y_info_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        else:
                            # Homography Failed
                            y_info_offset += 25
                            cv2.putText(display_frame, f"Homography/Inlier failed for {obj_name}", (10, y_info_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # --- UI/FPS/Key Handling ---
                
                # Show FPS
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if tracked_objects_count > 0:
                    cv2.putText(display_frame, f"TRACKING {tracked_objects_count} OBJECT(S) OK", (10, display_frame.shape[0]-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                
                cv2.imshow('Multi-Object Pose Estimation Test', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                
                elif key == ord('r'):  # Reset all
                    for obj_name in self.targets:
                        state = self.targets[obj_name]
                        state["calibrated"] = False
                        state["ref_image"] = None
                        state["ref_keypoints"] = None
                        state["ref_descriptors"] = None
                        state["rvec_smooth"] = None 
                        state["tvec_smooth"] = None
                        state["calib_buffer"].clear() # 🟢 Clears the buffer!                
                    print("✓ All calibrations reset. Place objects clearly in view for autocalibration.")
                
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
    print("✓ Using hardcoded camera calibration.")
    print("✓ 3D object points are centered at [0, 0, 0].")
    
    # UPDATE THIS PATH TO YOUR TRAINED MODEL
    yolo_model_path = '../../runs/detect/yolov8n_detect_V5/weights/best.pt'
    
    camera_index = 1
    
    # Instantiate the estimator object first
    estimator = MultiObjectPoseEstimator(yolo_model_path, camera_index)
    
    # Accessing attributes through the 'estimator' instance
    print("\n--- Object Configurations ---")
    for obj_name, state in estimator.targets.items():
        print(f"-> {obj_name.capitalize()}: {state['width_mm']}x{state['height_mm']}mm "
              f"(Min Matches: {state['min_matches']}, Min Inlier: {state['min_inlier_ratio']}, "
              f"EMA: {state['ema_alpha']}, Calib Buffer: {state['calib_buffer_size']})")                
    print("="*60 + "\n")

    estimator.run()


if __name__ == "__main__":
    main()