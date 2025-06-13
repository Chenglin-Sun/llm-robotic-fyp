#!/usr/bin/env python3
"""
realsense_aruco_pose.py
-----------------------
Detect ArUCo markers in real‑time using an Intel® RealSense™ RGB stream and
estimate their 6‑DoF pose (rotation and translation) with OpenCV.

* Requirements
  - Python 3.8+
  - pyrealsense2 (`pip install pyrealsense2`)
  - opencv‑python‑contrib ≥ 4.7 (`pip install opencv-contrib-python`)
  - numpy

* Usage
  python realsense_aruco_pose.py                 # default options
  python realsense_aruco_pose.py --type DICT_4X4_50 --marker_len 0.03
  python realsense_aruco_pose.py --record out/   # save annotated frames

Keys while running
  q  – quit
  r  – toggle recording of annotated frames
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ------------------------------ NEW ---------------------------------- #
FONT = cv2.FONT_HERSHEY_SIMPLEX          # consistent GUI font
TEXT_COLOR = (0, 255, 0)                 # green text
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# ArUCo dictionary lookup table                                               #
# --------------------------------------------------------------------------- #

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

# ----------------------------- USER PARAMS ---------------------------- #
BOARD_ROWS = 8          # number of squares vertically
BOARD_COLS = 8          # number of squares horizontally
# (optional) specify which marker IDs you glued to which corner.
#            If your IDs are different, edit this mapping.
CORNER_ID2NAME = {
    0: "TL",   # top-left
    1: "TR",   # top-right
    2: "BR",   # bottom-right
    3: "BL",   # bottom-left
}
ROBOT_MARKER_ID = 4 
TARGET_PERIOD = 0.1   # seconds   → 1 Hz

MARKER2ORIGIN_OFF = 0.079

ALPHA = 0.15     # smoothing factor  (0 → very slow, 1 → raw values)


# --- 1. compute / hard-code the four parameters once -----------
a, b = 0.94425768, 0.03883060   # s·cosθ , s·sinθ
tx, ty = 0.01371549, -0.0498331     # mm


# ------------------------------------------------------------------ #
#  NEW: global 3×3 homography  (mm → mm)
#         fitted on all test points (see report)
# ------------------------------------------------------------------ #
H_GRID = np.array([[ 0.921640, -0.088993, 16.296],
                   [ 0.035368,  0.938842, -9.251],
                   [-5.804e-05,-1.636e-04, 1.000000]],
                  dtype=np.float32)

def homography_correct(points_mm, H=H_GRID):
    """
    Apply 3×3 homography to an (...,2) array of (x,y) in mm.
    Returns an array of the same shape in mm.
    """
    shp   = points_mm.shape
    pts   = points_mm.reshape(-1, 2).astype(np.float32)
    ones  = np.ones((pts.shape[0], 1), np.float32)
    pts_h = np.hstack([pts, ones])                # (N,3)
    proj  = (H @ pts_h.T).T                      # (N,3)
    proj /= proj[:, 2:3]                         # divide by w
    return proj[:, :2].reshape(shp)


# ---------------------------------------------------------------------- #

# -----------------------  helper: order corners ----------------------- #
def order_corners(ids, img_corners, tvecs):
    """
    Returns TL, TR, BR, BL corners in both image-pixel and camera-metre space.
    img_corners : list[4x2]  : each marker's four pixel corners
    tvecs       : Nx3        : camera-space position of each marker centre
    """
    # map id -> index
    idx_map = {int(idx): i for i, idx in enumerate(ids.flatten())}
    missing = [cid for cid in CORNER_ID2NAME if cid not in idx_map]
    if missing:
        return None  # at least one corner not seen

    ordered_img = []
    ordered_cam = []
    for cid in [0, 1, 2, 3]:                       # TL, TR, BR, BL
        i = idx_map[cid]
        ordered_cam.append(tvecs[i].flatten())     # (x, y, z) in metres
        # we’ll take the *centre* of the marker for robustness
        centre_px = img_corners[i][0].mean(axis=0) # mean of 4 pts
        ordered_img.append(centre_px)
    return np.array(ordered_img), np.array(ordered_cam)
# ---------------------------------------------------------------------- #


# ---------------- helper: generate square centres -------------------- #
def board_centres_quad(cam_TL, cam_TR, cam_BR, cam_BL, rows, cols):
    """
    Bilinear-interpolated square centres from the *four* corner markers.

    Parameters
    ----------
    cam_TL, cam_TR, cam_BR, cam_BL : (3,) ndarray
        XYZ of the four markers in *camera* coordinates
        in Top-Left, Top-Right, Bottom-Right, Bottom-Left order.
    rows, cols : int
        Board dimensions (squares, not corners).

    Returns
    -------
    centres : (rows, cols, 3) ndarray
        Camera-space centre coordinates of every square.
    """
    TL, TR, BR, BL = map(np.asarray, (cam_TL, cam_TR, cam_BR, cam_BL))

    # pre-compute the four corner coefficients for each square
    r_idx, c_idx = np.indices((rows, cols), dtype=np.float32)
    s = (c_idx + 0.5) / cols        # horizontal parameter   (0 … 1)
    t = (r_idx + 0.5) / rows        # vertical parameter     (0 … 1)

    # bilinear blend:  P(s,t) = (1-s)(1-t)·TL + s(1-t)·TR + st·BR + (1-s)t·BL
    centres = ((1 - s)[..., None]*(1 - t)[..., None]*TL +
               s[..., None]*(1 - t)[..., None]*TR +
               s[..., None]*t[..., None]*BR +
               (1 - s)[..., None]*t[..., None]*BL)

    return centres.astype(np.float32)
# ---------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #

def intrinsics_to_opencv(intr):
    """Convert pyrealsense2 intrinsics to OpenCV camera matrix + distortion."""
    K = np.array(
        [[intr.fx, 0, intr.ppx],
         [0, intr.fy, intr.ppy],
         [0,       0,        1]],
        dtype=np.float32,
    )
    dist = np.array(intr.coeffs[:5], dtype=np.float32)
    return K, dist


def get_aruco_detector(dictionary_name: str):
    """Return an OpenCV ArUCo detector object + dictionary."""
    if dictionary_name not in ARUCO_DICT:
        raise KeyError(f"Unsupported ArUCo type '{dictionary_name}'.")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dictionary_name])
    parameters = cv2.aruco.DetectorParameters()

    # OpenCV >= 4.7 exposes the `ArucoDetector` class.  Fall back otherwise.
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        def detect(gray):
            return detector.detectMarkers(gray)

    except AttributeError:

        def detect(gray):
            return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    return aruco_dict, detect

def rvec_tvec_to_H(rvec, tvec):
    """Rodrigues rvec + 3×1 tvec  →  4×4 homogeneous transform."""
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4, dtype=np.float32)
    H[:3, :3] = R
    H[:3, 3] = tvec.flatten()
    return H

def transform_pts(H, pts):
    """
    Apply 4×4 transform H to (N,3) points → (N,3) points.
    """
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), np.float32)])   # → (N,4)
    return (H @ pts_h.T).T[:, :3]

def similarity_correct(centres_mm):
    """
    centres_mm : (...,2) array  (x,y in *mm*)
    returns     : (...,2) array  corrected to ground-truth frame
    """
    x, y = centres_mm[...,0], centres_mm[...,1]
    x_new = a*x - b*y + tx
    y_new = b*x + a*y + ty
    return np.stack([x_new, y_new], axis=-1)



# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
mapping_locked = False      # becomes True after user presses 'c'
centres_base_filtered = None

def main():
    parser = argparse.ArgumentParser(
        description="Real-time ArUCo pose estimation with Intel RealSense"
    )
    parser.add_argument("--type", default="DICT_4X4_50",  # fixed typo
                        help="ArUCo dictionary to use")
    parser.add_argument("--marker_len", type=float, default=0.028,
                        help="Physical side length of the marker in metres")
    parser.add_argument("--record", type=str, default=None, nargs="?",
                        const="captures",
                        help="(Optional) Directory to save annotated frames.")
    args = parser.parse_args()

    # ---------------- RealSense initialisation ----------------------- #
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile  = pipeline.start(config)
    intr     = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix, dist_coeffs = intrinsics_to_opencv(intr)

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coeffs.ravel())

    aruco_dict, detect = get_aruco_detector(args.type)
    marker_len         = float(args.marker_len)

    # --------------- Optional recording setup ------------------------ #
    recording = False
    if args.record:
        save_root  = Path(args.record)
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        save_dir   = save_root / f"session_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Info] Record directory: {save_dir.resolve()}")

    print("Press  r = record   c = lock mapping   q = quit")

    # ----------------------------- state ----------------------------- #
    mapping_locked        = False
    printed_once          = False
    centres_base_filtered = None

    # ----------------------------- MAIN LOOP ------------------------- #
    try:
        next_time = time.time()
        
        while True:
            # 1-Hz throttle
            if time.time() < next_time:
                pipeline.wait_for_frames()
                continue
            next_time += TARGET_PERIOD

            # -------- acquire frame -------- #
            frame = pipeline.wait_for_frames().get_color_frame()
            if not frame:
                continue
            color_img = np.asanyarray(frame.get_data())
            gray      = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            # -------- detect markers -------- #
            corners, ids, _ = detect(gray)
            if ids is None:
                cv2.imshow("RealSense ArUCo Pose", color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            cv2.aruco.drawDetectedMarkers(color_img, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_len, camera_matrix, dist_coeffs
            )

            # ---- axes + ID overlay ---- #
            for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                cv2.drawFrameAxes(color_img, camera_matrix, dist_coeffs,
                                  rvec, tvec, marker_len*0.5)
                tl = corners[idx][0][0]
                cv2.putText(color_img, str(int(ids[idx])),
                            (int(tl[0]), int(tl[1]) - 5),
                            FONT, 0.5, (0,255,255), 1, cv2.LINE_AA)

            # ---- chessboard pose (IDs 0-3) ---- #
            oc = order_corners(ids, corners, tvecs)
            if not oc:
                cv2.imshow("RealSense ArUCo Pose", color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue
            _, cam4 = oc
            TL, TR, BR, BL = cam4

            centres_cam = board_centres_quad(TL, TR, BR, BL,
                                        BOARD_ROWS, BOARD_COLS)

            # centres_cam = board_centres_pca(cam4,
            #                                      BOARD_ROWS,
            #                                      BOARD_COLS)



            # project to image (for red dots)
            proj, _  = cv2.projectPoints(centres_cam.reshape(-1,3),
                                         np.zeros(3), np.zeros(3),
                                         camera_matrix, dist_coeffs)
            proj = proj.reshape(BOARD_ROWS, BOARD_COLS, 2)

            # ---- robot-base marker (ID 4) ---- #
            idx_base = None
            for i, m in enumerate(ids.flatten()):
                if m == ROBOT_MARKER_ID:
                    idx_base = i
                    break

            if idx_base is not None:

                # -----------------------------------------------------------------
                R_marker, _ = cv2.Rodrigues(rvecs[idx_base])
                z_cam = R_marker[:, 2]                 # marker's +Z in camera frame

                # keep previous sign if the new sign is uncertain (|cos θ| small)
                if abs(z_cam[2]) < 0.2:                  # <≈11°
                    pose_flipped = last_pose_flipped      # re-use previous decision
                else:
                    pose_flipped =  z_cam[2] > 0          # true → backside seen

                last_pose_flipped = pose_flipped

                if pose_flipped:
                    R_marker = R_marker @ np.diag([1,-1,-1])
                    rvecs[idx_base], _ = cv2.Rodrigues(R_marker)

    
                H_base_cam = rvec_tvec_to_H(rvecs[idx_base], tvecs[idx_base])
                H_cam_base = np.linalg.inv(H_base_cam)
                base_tvec  = tvecs[idx_base].flatten()

                centres_base = transform_pts(
                    H_cam_base, centres_cam.reshape(-1,3)
                ).reshape(BOARD_ROWS, BOARD_COLS, 3)

                if centres_base_filtered is None:
                    centres_base_filtered = centres_base.copy()
                else:
                    centres_base_filtered = (
                        ALPHA * centres_base + (1-ALPHA) * centres_base_filtered
                    )

                centres_robot = centres_base_filtered[..., [2, 0, 1]]
                grid_offset = centres_robot.copy()
                grid_offset[..., 0] += MARKER2ORIGIN_OFF 
                grid_offset = grid_offset[..., :2]        # shape (8, 8, 2)  
                # --- after you have grid_offset (shape (8,8,2) in *metres*) ----------
                grid_mm      = grid_offset * 1000.0            # → mm
                grid_corr_mm = homography_correct(grid_mm)     # apply H
                grid_corr    = grid_corr_mm / 1000.0           # back to metres

                # ---------- FIXED drawMarker call ---------- #
                centre_px = corners[idx_base][0].mean(axis=0).astype(int)
                cv2.drawMarker(
                    color_img,
                    tuple(centre_px),            # position
                    (255, 0, 0),                 # colour  (BGR)
                    cv2.MARKER_CROSS,            # type
                    14,                          # markerSize  (int)
                    2                            # thickness (int)
                )
                # ------------------------------------------- #
            else:
                base_tvec = np.array([np.nan]*3, np.float32)

            # draw red dots + labels
            files = "ABCDEFGH"
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    x,y = proj[r,c]
                    cv2.circle(color_img, (int(x),int(y)), 3, (0,0,255), -1)
                    cv2.putText(color_img, f"{files[c]}{BOARD_ROWS-r}",
                                (int(x)+4,int(y)-4),
                                FONT, 0.4, (255,255,255),1,cv2.LINE_AA)

            # ---- key handling ---- #
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord("r") and args.record:
                recording = not recording
                print("Recording", "ON" if recording else "OFF")
            if k == ord("c"):
                mapping_locked = not mapping_locked
                print("Mapping", "LOCKED" if mapping_locked else "UNLOCKED")

            # ---- one-time print & save ---- #
            if mapping_locked and not printed_once and idx_base is not None:
                np.set_printoptions(precision=3, suppress=True)
                print("\n========== CALIBRATION RESULT ==========")
                print("Robot marker (camera):", base_tvec)
                print("\nGrid centres shifted (robot frame, metres):")
                print(grid_offset)
                print("\nGrid centres after homography, m:")
                print(grid_corr)
                # print("\nGrid centres shifted and corrected (robot frame, metres):")
                # print(grid_offset_corrected)
                out = (save_dir if args.record else Path(".")) / "final_annotation.png"
                cv2.imwrite(str(out), color_img)
                print("Annotated image →", out.resolve())
                print("=========================================\n")
                np.save("grid_robot_xy.npy", grid_offset)
                np.save("grid_robot_xy_homog.npy", grid_corr)

                # np.save("grid_robot_xy_corrected.npy", grid_offset_corrected)
                printed_once = True

            if recording and args.record:
                cv2.imwrite(str(save_dir / f"{int(time.time()*1000)}.png"), color_img)

            cv2.imshow("RealSense ArUCo Pose", color_img)

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == "__main__":
    main()
