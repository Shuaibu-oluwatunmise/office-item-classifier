"""
Camera Calibration using Checkerboard Pattern
Calculates camera intrinsic matrix and distortion coefficients
"""

import cv2
import numpy as np
import glob
import os
import json
from datetime import datetime


class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=25.0, camera_index=1):
        """
        Initialize camera calibrator
        
        Args:
            checkerboard_size: (cols, rows) - interior corners (not squares!)
                               For 10x7 squares checkerboard, use (9, 6)
            square_size: Size of each square in mm
            camera_index: Camera to use
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.camera_index = camera_index
        
        # Prepare object points (3D points in real world)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # Convert to mm
        
        # Storage for calibration
        self.obj_points = []  # 3D points in real world
        self.img_points = []  # 2D points in image
        self.images_used = []
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
        # Create output directory
        self.calib_dir = "camera_calibration"
        os.makedirs(self.calib_dir, exist_ok=True)
        os.makedirs(os.path.join(self.calib_dir, "images"), exist_ok=True)
    
    def capture_images(self, num_images=20):
        """
        Capture calibration images
        
        Args:
            num_images: Number of images to capture
        """
        print("\n" + "="*60)
        print("CAMERA CALIBRATION - IMAGE CAPTURE")
        print("="*60)
        print(f"\nCheckerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} interior corners")
        print(f"Square size: {self.square_size}mm")
        print(f"Target: {num_images} images")
        print("\nInstructions:")
        print("  - Hold checkerboard in different positions")
        print("  - Different angles and distances")
        print("  - Cover different parts of the frame")
        print("  - Press SPACE when pattern is detected (green corners)")
        print("  - Press ESC to finish early")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_index}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        count = 0
        
        try:
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = frame.copy()
                
                # Find checkerboard corners
                ret_corners, corners = cv2.findChessboardCorners(
                    gray, 
                    self.checkerboard_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                
                if ret_corners:
                    # Refine corner locations
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Draw corners
                    cv2.drawChessboardCorners(display_frame, self.checkerboard_size, 
                                             corners_refined, ret_corners)
                    
                    # Show status
                    cv2.putText(display_frame, "Pattern detected! Press SPACE to capture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Captured: {count}/{num_images}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Pattern not detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Captured: {count}/{num_images}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Camera Calibration - Capture', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print(f"\nCapture stopped by user. Captured {count} images.")
                    break
                elif key == ord(' ') and ret_corners:  # SPACE
                    # Save image
                    filename = os.path.join(self.calib_dir, "images", f"calib_{count:02d}.jpg")
                    cv2.imwrite(filename, frame)
                    
                    self.obj_points.append(self.objp)
                    self.img_points.append(corners_refined)
                    self.images_used.append(filename)
                    
                    count += 1
                    print(f"✓ Captured image {count}/{num_images}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if count < 10:
            print(f"\nWARNING: Only captured {count} images. Recommended minimum: 10")
            print("Calibration accuracy may be poor.")
            return False
        
        print(f"\n✓ Capture complete! {count} images saved.")
        return True
    
    def calibrate(self, image_size):
        """
        Perform camera calibration
        
        Args:
            image_size: (width, height) of images
        """
        print("\n" + "="*60)
        print("PERFORMING CALIBRATION")
        print("="*60)
        
        if len(self.obj_points) < 10:
            print("ERROR: Need at least 10 images for calibration!")
            return False
        
        print(f"Using {len(self.obj_points)} images...")
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            image_size,
            None,
            None
        )
        
        if not ret:
            print("ERROR: Calibration failed!")
            return False
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.obj_points)):
            img_points_projected, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.img_points[i], img_points_projected, cv2.NORM_L2) / len(img_points_projected)
            total_error += error
        
        self.calibration_error = total_error / len(self.obj_points)
        
        print("\n" + "="*60)
        print("✓ CALIBRATION SUCCESSFUL!")
        print("="*60)
        print(f"\nReprojection Error: {self.calibration_error:.4f} pixels")
        print("\nCamera Matrix (K):")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        print("="*60)
        
        return True
    
    def save_calibration(self):
        """Save calibration results to file"""
        if self.camera_matrix is None:
            print("ERROR: No calibration data to save!")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        calib_data = {
            "timestamp": timestamp,
            "checkerboard_size": self.checkerboard_size,
            "square_size_mm": self.square_size,
            "num_images": len(self.obj_points),
            "reprojection_error_pixels": float(self.calibration_error),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "images_used": self.images_used
        }
        
        json_file = os.path.join(self.calib_dir, f"camera_calibration_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(calib_data, f, indent=4)
        
        # Also save as Python file for easy import
        py_file = os.path.join(self.calib_dir, "camera_params.py")
        with open(py_file, 'w') as f:
            f.write(f"# Camera Calibration Parameters\n")
            f.write(f"# Generated: {timestamp}\n")
            f.write(f"# Reprojection Error: {self.calibration_error:.4f} pixels\n\n")
            f.write(f"import numpy as np\n\n")
            f.write(f"CAMERA_MATRIX = np.array([\n")
            for row in self.camera_matrix:
                f.write(f"    {list(row)},\n")
            f.write(f"], dtype=np.float32)\n\n")
            f.write(f"DIST_COEFFS = np.array({self.dist_coeffs.flatten().tolist()}, dtype=np.float32)\n")
        
        print(f"\n✓ Calibration saved:")
        print(f"   JSON: {json_file}")
        print(f"   Python: {py_file}")
        
        return True
    
    def test_undistortion(self):
        """Test undistortion on captured images"""
        if self.camera_matrix is None:
            print("ERROR: No calibration data!")
            return
        
        print("\n" + "="*60)
        print("TESTING UNDISTORTION")
        print("="*60)
        print("Press any key to see next image, ESC to exit")
        print("="*60 + "\n")
        
        for img_file in self.images_used[:5]:  # Show first 5 images
            img = cv2.imread(img_file)
            h, w = img.shape[:2]
            
            # Get optimal new camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            
            # Undistort
            undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, 
                                       None, new_camera_matrix)
            
            # Crop if needed
            x, y, w_roi, h_roi = roi
            if w_roi > 0 and h_roi > 0:
                undistorted = undistorted[y:y+h_roi, x:x+w_roi]
            
            # Show comparison
            comparison = np.hstack([img, undistorted])
            comparison = cv2.resize(comparison, (1280, 480))
            
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Undistorted", (650, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Undistortion Test', comparison)
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        print("✓ Test complete")
    
    def run(self, num_images=20):
        """Run complete calibration workflow"""
        print("\n" + "="*70)
        print(" " * 20 + "CAMERA CALIBRATION")
        print("="*70)
        
        # Step 1: Capture images
        if not self.capture_images(num_images):
            print("\n❌ Calibration failed: Not enough images captured")
            return False
        
        # Get image size from first captured image
        first_img = cv2.imread(self.images_used[0])
        h, w = first_img.shape[:2]
        
        # Step 2: Calibrate
        if not self.calibrate((w, h)):
            print("\n❌ Calibration failed: Calibration error")
            return False
        
        # Step 3: Save results
        if not self.save_calibration():
            print("\n❌ Failed to save calibration")
            return False
        
        # Step 4: Test undistortion
        self.test_undistortion()
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Results saved in: {self.calib_dir}/")
        print(f"📊 Reprojection error: {self.calibration_error:.4f} pixels")
        print(f"   (Lower is better. Good: < 0.5, Acceptable: < 1.0)")
        print("\n💡 Next steps:")
        print("   1. Check camera_calibration/camera_params.py")
        print("   2. Use these values in your pose estimation script")
        print("="*70 + "\n")
        
        return True


def main():
    print("\n" + "="*70)
    print("CAMERA CALIBRATION SETUP")
    print("="*70)
    print("\n📋 Checkerboard Information:")
    print("   - Count the INTERIOR corners (not squares!)")
    print("   - For a 10x7 squares board → Use (9, 6)")
    print("   - For a 9x6 squares board → Use (8, 5)")
    print("\n📏 Square Size:")
    print("   - Measure one square with a ruler")
    print("   - Common sizes: 20mm, 25mm, 30mm")
    print("\n🎥 Camera:")
    print("   - Make sure camera index is correct")
    print("   - Default is 1 (external webcam)")
    print("="*70 + "\n")
    
    # CONFIGURE THESE VALUES FOR YOUR CHECKERBOARD
    checkerboard_size = (9, 6)  # (cols, rows) of INTERIOR corners
    square_size = 24.0          # Size of one square in mm
    camera_index = 1            # Camera to use
    num_images = 20             # Number of calibration images
    
    print(f"Configuration:")
    print(f"  Checkerboard: {checkerboard_size[0]}x{checkerboard_size[1]} interior corners")
    print(f"  Square size: {square_size}mm")
    print(f"  Camera: {camera_index}")
    print(f"  Images to capture: {num_images}")
    print("\nPress ENTER to start or Ctrl+C to cancel...")
    input()
    
    calibrator = CameraCalibrator(checkerboard_size, square_size, camera_index)
    calibrator.run(num_images)


if __name__ == "__main__":
    main()