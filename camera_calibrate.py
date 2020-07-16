import numpy as np
import cv2
import os


CHESSBOARD_SIZE = (6, 9)


def calibrate(chessboard_path, show_chessboard=False):
    # Logical coordinates of chessboard corners
    obj_p = np.zeros((1, CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    obj_p[0, :, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    obj_points = []     # 3d point in real world space
    img_points = []     # 2d points in image plane.

    # Iterate through all images in the folder
    image_list = os.listdir(chessboard_path)
    gray = None
    for image in image_list:
        img = cv2.imread(os.path.join(chessboard_path, image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            # Refining corners position with sub-pixels based algorithm
            obj_points.append(obj_p)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            img_points.append(corners)
            print('Image ' + image + ' is valid for calibration')
            if show_chessboard:
                cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
                cv2.imwrite(os.path.join('./Chessboards_Corners', image), img)

    k = np.zeros((3, 3))
    d = np.zeros((4, 1))
    dims = gray.shape[::-1]
    num_valid_img = len(obj_points)
    if num_valid_img > 0:
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(num_valid_img)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(num_valid_img)]
        rms, _, _, _, _ = cv2.fisheye.calibrate(obj_points, img_points, gray.shape[::-1], k, d, rvecs, tvecs,
                                                # cv2.fisheye.CALIB_CHECK_COND +
                                                # When CALIB_CHECK_COND is set, the algorithm checks if the detected corners of each images are valid.
                                                # If not, an exception is thrown which indicates the zero-based index of the invalid image.
                                                # Such image should be replaced or removed from the calibration dataset to ensure a good calibration.
                                                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                                                cv2.fisheye.CALIB_FIX_SKEW,
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    print("Found " + str(num_valid_img) + " valid images for calibration")
    return k, d, dims


if __name__ == '__main__':
    if not os.path.exists('./parameters'):
        os.makedirs('./parameters')
    if not os.path.exists('./Chessboards_Corners'):
        os.makedirs('./Chessboards_Corners')

    K, D, Dims = calibrate('./Chessboards', show_chessboard=True)
    np.save('./parameters/Dims', np.array(Dims))
    np.save('./parameters/K', np.array(K))
    np.save('./parameters/D', np.array(D))
