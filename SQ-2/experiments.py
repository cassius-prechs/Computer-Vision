import os
import glob
import random
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Settings (keep in sync with calibrate.py)
CHECKERBOARD = (7, 7)
SQUARE_SIZE = 25.0
OUTPUT_DIR = "outputs_exp"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_corners_all(image_glob="images/*.jpg"):
    paths = sorted(glob.glob(image_glob))
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2))
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []
    images = []
    filenames = []
    image_size = None

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        # try the robust SB detector first
        ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, None)

        if not ret:
            print(f"No corners: {p}")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        images.append(img)
        filenames.append(os.path.basename(p))

    return {
        "paths": paths,
        "objpoints": objpoints,
        "imgpoints": imgpoints,
        "images": images,
        "filenames": filenames,
        "image_size": image_size,
    }


def calibrate_and_reproj(objpoints, imgpoints, image_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    # compute mean reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error = mean_error / len(objpoints)
    return {
        "ret": ret,
        "mtx": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "reproj_error": mean_error,
    }


def run_experiments(trials_per_N=5, Ns=None):
    data = detect_corners_all()
    M = len(data["objpoints"]) 
    if M == 0:
        print("No valid images with detected corners. Abort.")
        return

    if Ns is None:
        Ns = sorted(list(set([3, 5, 8, 10, 15, 20] + [M])))
    Ns = [n for n in Ns if n <= M]

    rows = []

    for n in Ns:
        for t in range(trials_per_N):
            idx = random.sample(range(M), n)
            sel_obj = [data["objpoints"][i] for i in idx]
            sel_img = [data["imgpoints"][i] for i in idx]
            r = calibrate_and_reproj(sel_obj, sel_img, data["image_size"]) 
            # pose diversity metric: std of rotation vector norms
            rnorms = [np.linalg.norm(rv.ravel()) for rv in r["rvecs"]]
            pose_diversity = float(np.std(rnorms)) if rnorms else 0.0

            rows.append({
                "N": n,
                "trial": t,
                "reproj_error": float(r["reproj_error"]),
                "pose_diversity": pose_diversity,
            })
            print(f"N={n} trial={t} reproj={r['reproj_error']:.4f} pose_div={pose_diversity:.4f}")

    # save CSV
    csv_path = os.path.join(OUTPUT_DIR, "experiments.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "trial", "reproj_error", "pose_diversity"])
        w.writeheader()
        w.writerows(rows)

    # aggregate and plot
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("N").agg({"reproj_error": ["mean", "std"], "pose_diversity": ["mean"]})
    agg.columns = ["_" .join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    fig, ax = plt.subplots()
    ax.errorbar(agg["N"], agg["reproj_error_mean"], yerr=agg["reproj_error_std"], marker="o")
    ax.set_xlabel("Number of images (N)")
    ax.set_ylabel("Mean reprojection error (px)")
    ax.set_title("Calibration error vs number of images")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "error_vs_N.png"), dpi=150)
    plt.close(fig)

    # simple pose diversity plot
    fig, ax = plt.subplots()
    ax.plot(agg["N"], agg["pose_diversity_mean"], marker="o")
    ax.set_xlabel("Number of images (N)")
    ax.set_ylabel("Pose diversity (std of rotvec norm)")
    ax.set_title("Pose diversity vs N")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "pose_diversity_vs_N.png"), dpi=150)
    plt.close(fig)

    print(f"Saved CSV to {csv_path} and plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiments(trials_per_N=5)
