import cv2
import numpy as np
import glob
import os

# Settings
# internal corners (if squares are 8x8, internal corners are 7x7)
CHECKERBOARD = (7, 7)

# square size [mm]
SQUARE_SIZE = 25.0

# output directory
OUTPUT_DIR = "outputs"

# draw a 3D box on each image
DRAW_3D = True
BOX_HEIGHT_SQUARES = 3

# 35mm-equivalent focal length from specs (mm); set to None to skip comparison
FOCAL_LENGTH_EQ_MM = 24.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prepare 3D points
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = (
    np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)
)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []
valid_images = []
valid_filenames = []
image_size = None
detection_results = []

# Load images
images = glob.glob("images/*.jpg")

print(f"Found {len(images)} images")

# Detect chessboard corners
for fname in images:

    img = cv2.imread(fname)

    if img is None:
        print(f"Cannot load {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        None
    )

    print(fname, ret)


    # log detection result for diagnostics
    basename = os.path.basename(fname)
    if ret and corners is not None:
        num_corners = int(corners.shape[0])
    else:
        num_corners = 0
    detection_results.append((basename, int(bool(ret)), num_corners))

    if ret:

        objpoints.append(objp)
        imgpoints.append(corners)
        valid_images.append(img)
        valid_filenames.append(os.path.basename(fname))

        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners,
            ret
        )
        
        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"corners_{basename}"
            ),
            img
        )

# write detection summary for all images (detected / not detected)
with open(os.path.join(OUTPUT_DIR, "detection_summary.csv"), "w") as f:
    f.write("filename,detected,num_corners\n")
    for name, det, nc in detection_results:
        f.write(f"{name},{det},{nc}\n")

if not objpoints:
    raise SystemExit("No chessboard corners detected. Check CHECKERBOARD and images.")

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\n============================")
print("Camera Matrix")
print("============================")
print(mtx)

print("\n============================")
print("Distortion Coefficients")
print("============================")
print(dist)

print("\n============================")
print("Extrinsic Parameters (R, t)")
print("============================")

for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    rmat, _ = cv2.Rodrigues(rvec)
    print(f"Image {i:02d} R =\n{rmat}")
    print(f"Image {i:02d} t =\n{tvec}\n")

if DRAW_3D:
    board_w = (CHECKERBOARD[0] - 1) * SQUARE_SIZE
    board_h = (CHECKERBOARD[1] - 1) * SQUARE_SIZE
    box_h = BOX_HEIGHT_SQUARES * SQUARE_SIZE

    box_points = np.float32(
        [
            [0, 0, 0],
            [board_w, 0, 0],
            [board_w, board_h, 0],
            [0, board_h, 0],
            [0, 0, -box_h],
            [board_w, 0, -box_h],
            [board_w, board_h, -box_h],
            [0, board_h, -box_h],
        ]
    )

    for i, (img, rvec, tvec) in enumerate(zip(valid_images, rvecs, tvecs)):
        imgpts, _ = cv2.projectPoints(
            box_points,
            rvec,
            tvec,
            mtx,
            dist
        )

        imgpts = np.int32(imgpts).reshape(-1, 2)

        cv2.polylines(img, [imgpts[:4]], True, (0, 255, 0), 2)
        cv2.polylines(img, [imgpts[4:]], True, (0, 255, 0), 2)

        for j in range(4):
            cv2.line(img, tuple(imgpts[j]), tuple(imgpts[j + 4]), (0, 255, 0), 2)

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"box_{valid_filenames[i]}"
            ),
            img
        )

# Reprojection error
mean_error = 0

for i in range(len(objpoints)):

    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        mtx,
        dist
    )

    error = cv2.norm(
        imgpoints[i],
        imgpoints2,
        cv2.NORM_L2
    ) / len(imgpoints2)

    mean_error += error

print("\n============================")
print("Mean Reprojection Error")
print("============================")
print(mean_error / len(objpoints))

# per-image reprojection errors and save CSV
per_errs = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        mtx,
        dist
    )
    err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    name = valid_filenames[i] if i < len(valid_filenames) else f"img_{i}"
    per_errs.append((name, float(err)))

with open(os.path.join(OUTPUT_DIR, "per_image_errors.csv"), "w") as f:
    f.write("filename,error\n")
    for name, e in per_errs:
        f.write(f"{name},{e:.6f}\n")

per_errs_sorted = sorted(per_errs, key=lambda x: x[1], reverse=True)
print("\nPer-image reprojection errors (px), largest first:")
for name, e in per_errs_sorted[:10]:
    print(f"{name}: {e:.4f}")

if FOCAL_LENGTH_EQ_MM is not None:
    image_width_px = image_size[0]
    f_px_from_spec = (FOCAL_LENGTH_EQ_MM / 36.0) * image_width_px
    f_px_est = 0.5 * (mtx[0, 0] + mtx[1, 1])

    print("\n============================")
    print("Focal Length Comparison")
    print("============================")
    print(f"Estimated f (px): {f_px_est:.2f}")
    print(f"From spec (px):   {f_px_from_spec:.2f}")

# Save parameters
np.save(os.path.join(OUTPUT_DIR, "camera_matrix.npy"), mtx)
np.save(os.path.join(OUTPUT_DIR, "dist_coeffs.npy"), dist)

print("\nSaved calibration results.")