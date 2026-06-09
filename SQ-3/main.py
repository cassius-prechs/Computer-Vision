import os
import cv2
import numpy as np

from feature_matching import build_pair_matches, assign_track_indices, match_features
from motion_models import (
    estimate_translation,
    estimate_similarity,
    estimate_affine,
    estimate_homography
)
from stitching import stitch_images

MODELS = {
    "translation": estimate_translation,
    "similarity": estimate_similarity,
    "affine": estimate_affine,
    "homography": estimate_homography
}

OUTPUT_DIR = "./outputs"


def load_images():
    images = {}
    for name in ["00", "01", "02", "10", "11", "12"]:
        img = cv2.imread(f"images/{name}.jpg")
        if img is None:
            raise FileNotFoundError(f"images/{name}.jpg")
        images[name] = img
    return images

def save_input_grid(images):
    row1 = np.hstack([images["00"], images["01"], images["02"]])
    row2 = np.hstack([images["10"], images["11"], images["12"]])
    grid = np.vstack([row1, row2])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "input_grid.jpg"), grid)

def save_matches(images):
    img1 = images["01"]
    img2 = images["11"]
    
    pts1, pts2 = match_features(img1, img2)
    
    vis = img1.copy()
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    for p1, p2 in zip(pts1[:100], pts2[:100]):
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]) + w1, int(p2[1]))
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(vis, pt1, 3, (0, 0, 255), -1)
        cv2.circle(vis, pt2, 3, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "matches.jpg"), vis)

def build_global_transforms(matches, estimator):
    T00_01 = estimator(*matches[("00","01")])
    T01_02 = estimator(*matches[("01","02")])
    T10_11 = estimator(*matches[("10","11")])
    T11_12 = estimator(*matches[("11","12")])
    T01_11 = estimator(*matches[("01","11")])

    transforms = {}
    transforms["11"] = np.eye(3, dtype=np.float32)

    transforms["10"] = T10_11                   
    transforms["12"] = np.linalg.inv(T11_12)    

    transforms["01"] = T01_11                   
    transforms["00"] = T00_01 @ T01_11          
    transforms["02"] = np.linalg.inv(T01_02) @ T01_11 

    return transforms

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = load_images()
    save_input_grid(images)
    save_matches(images)

    matches = build_pair_matches(images)
    tracks = assign_track_indices(matches)

    print(f"\nTotal tracks assigned: {len(tracks)}")

    track_path = os.path.join(OUTPUT_DIR, "tracks.csv")
    with open(track_path, "w") as f:
        f.write("track_id,image,x,y\n")
        for tid, obs in tracks.items():
            for img_name, (x, y) in obs.items():
                f.write(f"{tid},{img_name},{x:.2f},{y:.2f}\n")

    print(f"saved {track_path}")

    for model in ["translation", "similarity", "affine", "homography"]:
        estimator = MODELS[model]
        transforms = build_global_transforms(matches, estimator)
        result = stitch_images(images, transforms, model)

        output_path = os.path.join(OUTPUT_DIR, f"panorama_{model}.jpg")
        cv2.imwrite(output_path, result)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
