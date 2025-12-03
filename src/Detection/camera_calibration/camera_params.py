"""Camera calibration parameters for Raspberry Pi CSI Camera"""
import numpy as np

# Calibrated on 20251203_124346
# Resolution: 640x480
# Reprojection error: 0.2830 pixels
# Camera: Raspberry Pi CSI with libcamera

CAMERA_MATRIX = np.array([
    [657.154416539939, 0.0, 482.66896030538066],
    [0.0, 664.2379240609116, 213.47298599781223],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

DIST_COEFFS = np.array([-0.4523396208850756, -0.0049281188761248, 0.003955998278078904, -0.05678037089531683, 0.11416682400905645], dtype=np.float32)
