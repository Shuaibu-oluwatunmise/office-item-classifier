"""
Camera Calibration for Raspberry Pi CSI Camera
Uses libcamera + GStreamer pipeline
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np
import json
from datetime import datetime


class CSICameraCalibrator:
    def __init__(self, checkerboard_size, square_size):
        """
        Args:
            checkerboard_size: (width, height) in INTERIOR corners
            square_size: Size of one square in mm
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare 3D object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage
        self.obj_points = []
        self.img_points = []
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
        # GStreamer pipeline
        self.pipeline = None
        self.sink = None
        
        # Create output directory
        os.makedirs('camera_calibration/images', exist_ok=True)
    
    def start_camera(self):
        """Initialize CSI camera with GStreamer"""
        print("Initializing CSI camera...")
        
        Gst.init(None)
        
        # Use 640x480 resolution for calibration (match tracking resolution)
        self.pipeline = Gst.parse_launch(
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )
        
        self.sink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        print("✓ CSI camera started (640x480)")
        return True
    
    def pull_frame(self, timeout_ns=10_000_000):
        """Get frame from GStreamer pipeline"""
        sample = self.sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return None
        
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value("width")
        h = caps.get_value("height")
        
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3)
            return frame
        finally:
            buf.unmap(mapinfo)
    
    def capture_images(self, num_images=20):
        """Capture calibration images"""
        print("\n" + "="*60)
        print("CAMERA CALIBRATION - IMAGE CAPTURE")
        print("="*60)
        print(f"\nCheckerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} interior corners")
        print(f"Square size: {self.square_size}mm")
        print(f"Target: {num_images} images\n")
        print("Instructions:")
        print("  - Hold checkerboard in different positions")
        print("  - Different angles and distances")
        print("  - Cover edges, corners, center of frame")
        print("  - Press SPACE when GREEN corners appear")
        print("  - Press ESC to finish")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return False
        
        cv2.namedWindow("CSI Calibration", cv2.WINDOW_NORMAL)
        
        count = 0
        
        try:
            while count < num_images:
                frame = self.pull_frame()
                
                if frame is None:
                    # No frame yet, just wait
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = frame.copy()
                
                # Find checkerboard corners
                ret_corners, corners = cv2.findChessboardCorners(
                    gray, self.checkerboard_size, None
                )
                
                if ret_corners:
                    # Refine corner positions
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Draw corners
                    cv2.drawChessboardCorners(display, self.checkerboard_size, corners_refined, ret_corners)
                    
                    cv2.putText(display, f"Pattern OK! Press SPACE ({count}/{num_images})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display, f"No pattern detected ({count}/{num_images})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("CSI Calibration", display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nCapture cancelled")
                    break
                elif key == 32 and ret_corners:  # SPACE
                    # Save points
                    self.obj_points.append(self.objp)
                    self.img_points.append(corners_refined)
                    
                    # Save image
                    img_path = f'camera_calibration/images/calib_{count:02d}.jpg'
                    cv2.imwrite(img_path, frame)
                    
                    count += 1
                    print(f"✓ Captured image {count}/{num_images}")
        
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
        
        print(f"\n✓ Captured {count} images")
        return count >= 10  # Need at least 10 for calibration
    
    def calibrate(self, image_shape):
        """Perform camera calibration"""
        print("\n" + "="*60)
        print("PERFORMING CALIBRATION")
        print("="*60)
        print(f"Using {len(self.obj_points)} images...")
        
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, image_shape, None, None
        )
        
        if not ret:
            print("ERROR: Calibration failed!")
            return False
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(self.obj_points)):
            img_points_reproj, _ = cv2.projectPoints(
                self.obj_points[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
            mean_error += error
        
        mean_error /= len(self.obj_points)
        
        print("\n" + "="*60)
        print("✓ CALIBRATION SUCCESSFUL!")
        print("="*60)
        print(f"\nReprojection Error: {mean_error:.4f} pixels")
        print(f"\nCamera Matrix (K):")
        print(self.camera_matrix)
        print(f"\nDistortion Coefficients:")
        print(self.dist_coeffs.ravel())
        print("="*60)
        
        return True
    
    def save_calibration(self):
        """Save calibration to JSON and Python file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate mean reprojection error
        mean_error = 0
        for i in range(len(self.obj_points)):
            img_points_reproj, _ = cv2.projectPoints(
                self.obj_points[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
            mean_error += error
        mean_error /= len(self.obj_points)
        
        # Save to JSON
        json_path = f'camera_calibration/camera_calibration_{timestamp}.json'
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'rvecs': [r.tolist() for r in self.rvecs],
            'tvecs': [t.tolist() for t in self.tvecs],
            'reprojection_error': mean_error,
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size,
            'resolution': '640x480',
            'camera_type': 'Raspberry Pi CSI',
            'timestamp': timestamp
        }
        
        with open(json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n✓ Calibration saved:")
        print(f"   JSON: {json_path}")
        
        # Save to Python file for easy import
        py_path = 'camera_calibration/camera_params.py'
        with open(py_path, 'w') as f:
            f.write('"""Camera calibration parameters for Raspberry Pi CSI Camera"""\n')
            f.write('import numpy as np\n\n')
            f.write(f'# Calibrated on {timestamp}\n')
            f.write(f'# Resolution: 640x480\n')
            f.write(f'# Reprojection error: {mean_error:.4f} pixels\n')
            f.write(f'# Camera: Raspberry Pi CSI with libcamera\n\n')
            f.write('CAMERA_MATRIX = np.array([\n')
            for row in self.camera_matrix:
                f.write(f'    {row.tolist()},\n')
            f.write('], dtype=np.float32)\n\n')
            f.write(f'DIST_COEFFS = np.array({self.dist_coeffs.ravel().tolist()}, dtype=np.float32)\n')
        
        print(f"   Python: {py_path}")
        print("="*60 + "\n")
    
    def run(self, num_images=20):
        """Run full calibration workflow"""
        print("\n" + "="*60)
        print("RASPBERRY PI CSI CAMERA CALIBRATION")
        print("="*60)
        print(f"\n📋 Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} interior corners")
        print(f"📏 Square size: {self.square_size}mm")
        print(f"📸 Images to capture: {num_images}")
        print(f"🎥 Camera: CSI (libcamera + GStreamer)")
        print("="*60)
        
        input("\nPress ENTER to start or Ctrl+C to cancel...")
        
        # Capture images
        if not self.capture_images(num_images):
            print("ERROR: Not enough images captured")
            return
        
        # Get image shape from first saved image
        first_img = cv2.imread('camera_calibration/images/calib_00.jpg', cv2.IMREAD_GRAYSCALE)
        if first_img is None:
            print("ERROR: Cannot read calibration images")
            return
        
        image_shape = first_img.shape[::-1]  # (width, height)
        
        # Calibrate
        if not self.calibrate(image_shape):
            return
        
        # Save results
        self.save_calibration()
        
        print("✓ Calibration complete!")
        print("\nNext steps:")
        print("  1. Check reprojection error (should be < 0.5 pixels)")
        print("  2. Run pose tracking with: python3 pi_csi_pose_tracker.py")


def main():
    # YOUR CHECKERBOARD PARAMETERS
    checkerboard_size = (9, 6)  # 10x7 squares = 9x6 interior corners
    square_size = 24.0          # 24mm per square
    num_images = 20
    
    calibrator = CSICameraCalibrator(checkerboard_size, square_size)
    calibrator.run(num_images)


if __name__ == "__main__":
    main()