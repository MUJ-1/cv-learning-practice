import cv2
import numpy as np
import os

# ────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────

FOLDER = "classical_cv/camera_callibration/images_aruco"
MARKER_SIZE_METERS = 0.15          # 15 cm marker
MIN_IMAGES_FOR_CALIB = 10

# List of dictionaries to try (most common ones first)
DICTIONARIES_TO_TRY = [
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_6X6_100,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_5X5_250,
    cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_4X4_1000,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_7X7_250,
    cv2.aruco.DICT_ARUCO_ORIGINAL,
    cv2.aruco.DICT_APRILTAG_36h11,
    # add more only if needed — these cover most printed markers
]

# ────────────────────────────────────────────────
#  Find images
# ────────────────────────────────────────────────

image_paths = [
    os.path.join(FOLDER, fname)
    for fname in os.listdir(FOLDER)
    if fname.lower().endswith(('.jpeg', '.jpg', '.png'))
]

image_paths.sort()  # nicer output order

print(f"Found {len(image_paths)} images:")
for p in image_paths:
    print("  ", p)
print()

if not image_paths:
    raise FileNotFoundError("No images found in folder")

# ────────────────────────────────────────────────
#  Auto-detect which dictionary is being used
# ────────────────────────────────────────────────

test_img = cv2.imread(image_paths[0])
if test_img is None:
    raise IOError(f"Cannot read test image: {image_paths[0]}")

gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

detector = None
dict_name = None

for dict_id in DICTIONARIES_TO_TRY:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    # You can tune parameters here if detection is unreliable
    # params.adaptiveThreshConstant = 7
    # params.minMarkerPerimeterRate = 0.03
    temp_detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = temp_detector.detectMarkers(gray_test)

    if ids is not None and len(ids) > 0:
        detector = temp_detector
        dict_name = dict_id
        print(f"→ Dictionary found: {dict_id} (detected marker in first image)")
        break

if detector is None:
    raise RuntimeError(
        "Could not detect any marker in the first image with any dictionary.\n"
        "Possible causes:\n"
        "  • Marker is not an ArUco / AprilTag\n"
        "  • Image is too blurry / poorly lit / too small\n"
        "  • Marker was printed incorrectly (wrong size, inverted, etc.)"
    )

# ────────────────────────────────────────────────
#  Prepare calibration data
# ────────────────────────────────────────────────

# 3D model points of one marker (center at origin)
half = MARKER_SIZE_METERS / 2
objp = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float32)

all_obj_points = []
all_img_points = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"  [SKIP] Cannot read {os.path.basename(path)}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 1:
        # Take the first detected marker (you should have only one per image)
        all_img_points.append(corners[0])      # (1,4,2) → we take [0]
        all_obj_points.append(objp)
        print(f"  [ OK ] {os.path.basename(path)}")
    else:
        print(f"  [FAIL] No marker detected in {os.path.basename(path)}")

print()
print(f"Usable images for calibration: {len(all_img_points)} / {len(image_paths)}")

if len(all_img_points) < MIN_IMAGES_FOR_CALIB:
    print(f"⚠️  Warning: Fewer than {MIN_IMAGES_FOR_CALIB} good images → result may be poor")
# ────────────────────────────────────────────────
#  Run calibration
# ────────────────────────────────────────────────

if len(all_img_points) < 4:
    print("Too few images → cannot calibrate.")
else:
    h, w = gray.shape   # from last processed image — assume all same resolution
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points,
        all_img_points,
        (w, h),
        None, None
    )

    print("\n" + "="*60)
    if ret:
        print("Calibration succeeded")
    else:
        print("Calibration FAILED (ret == False)")
    print("="*60)

    print("Camera matrix:\n", np.round(mtx, 4))
    print("\nDistortion coeffs:\n", np.round(dist.ravel(), 6))
    print()

    # ─── Reprojection error ───────────────────────────────────────────────
mean_error = 0.0
n_good = len(all_obj_points)

for i in range(n_good):
    imgpts2, _ = cv2.projectPoints(
        all_obj_points[i],
        rvecs[i], tvecs[i],
        mtx, dist
    )
    # Make both (N,2) float32
    projected = imgpts2.reshape(-1, 2).astype(np.float32)
    observed  = all_img_points[i].astype(np.float32)

    # Manual L2 norm per point average (avoids type/channel mismatch)
    diff = observed - projected
    error = np.sqrt(np.sum(diff * diff)) / len(projected)   # == cv2.NORM_L2 / N

    mean_error += error

mean_error /= n_good

print(f"Mean reprojection error: {mean_error:.4f} pixels")
print("  → Good values usually 0.1–0.8 px for clean data; >2 px means issues")