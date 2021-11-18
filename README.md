# fisheye-camera-undistortion
Fisheye camera distortion correction based on opencv's chessboard calibration algorithm
### Environment

- python==3.6.5
- opencv-python==4.2.0


### Getting Started

1. Take several standard chessboard images with the fisheye lens to be corrected and placed into the 'Chessboards' folder (12 chessboard images taken with the lens used for the test image have been placed).
2. Run 'camera_calibrate.py' to calculate the internal parameter matrix K and the distortion coefficient vector D.

3. Run 'image_correction.py' to correct a single image captured by the camera.

4. Run 'video_correction.py' to correct the camera in real time.
