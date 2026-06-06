import cv2
import numpy as np
import os

# Load Images
os.makedirs("./outputs", exist_ok=True)
img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect Features (SIFT)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Feature Matching (BFMatcher + Lowe's Ratio Test, threshold=0.75)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(f"Good Matches: {len(good)}")

# Draw Matching Result
match_img = cv2.drawMatches(
    img1, kp1, img2, kp2, good, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("./outputs/matches.jpg", match_img)

# Extract Corresponding Points (float32 for subpixel accuracy)
pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# Estimate Fundamental Matrix (RANSAC, threshold=3.0px, confidence=0.99)
F, mask = cv2.findFundamentalMat(
    pts1, pts2,
    cv2.FM_RANSAC,
    ransacReprojThreshold=3.0,
    confidence=0.99
)

print("Fundamental Matrix:")
print(F)
print(f"Inliers: {int(mask.sum())} / {len(good)}")

# Verify rank-2 constraint: det(F) should be ~0
print(f"det(F) = {np.linalg.det(F):.6e}  (should be ~0)")

# Inlier points only
pts1_in = pts1[mask.ravel() == 1]
pts2_in = pts2[mask.ravel() == 1]

# Quantitative Evaluation: Symmetric Epipolar Distance
def compute_epipolar_distances(pts1, pts2, F):
    """
    Compute symmetric epipolar distance for each point pair.
    d_sym = (|x'^T F x| / ||Fx||_perp + |x'^T F x| / ||F^T x'||_perp) / 2
    """
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])  # (N, 3) homogeneous
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])

    lines2 = (F   @ pts1_h.T).T   # epipolar lines in image2: shape (N, 3)
    lines1 = (F.T @ pts2_h.T).T   # epipolar lines in image1: shape (N, 3)

    def point_line_dist(pts_h, lines):
        num = np.abs(np.sum(pts_h * lines, axis=1))
        den = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2) + 1e-10
        return num / den

    d1 = point_line_dist(pts1_h, lines1)
    d2 = point_line_dist(pts2_h, lines2)
    return (d1 + d2) / 2

dists = compute_epipolar_distances(pts1_in, pts2_in, F)
print(f"Symmetric epipolar distance (inliers): mean={dists.mean():.4f} px, std={dists.std():.4f} px")

# Function to Draw Epipolar Lines
def draw_epilines(img_lines, img_points, lines, pts_on_lines, pts_on_points):
    """
    Draw epipolar lines and corresponding points.
    - img_lines  : image on which epilines are drawn
    - img_points : image on which corresponding points are drawn
    - lines      : epipolar lines (N, 3), each row [a, b, c] for ax+by+c=0
    """
    h, w = img_lines.shape[:2]
    img_lines  = img_lines.copy()
    img_points = img_points.copy()

    for line_coef, pt_line, pt_point in zip(lines, pts_on_lines, pts_on_points):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        a, b, c = line_coef
        x0, y0 = 0,  int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)

        cv2.line(img_lines, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img_lines,  tuple(pt_line.astype(int)),  6, color, -1)
        cv2.circle(img_points, tuple(pt_point.astype(int)), 6, color, -1)

    return img_lines, img_points

# Epilines in image1 (from points in image2)
lines1 = cv2.computeCorrespondEpilines(pts2_in.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
left_with_epilines, right_with_dots = draw_epilines(img1, img2, lines1, pts1_in, pts2_in)

# Epilines in image2 (from points in image1)
lines2 = cv2.computeCorrespondEpilines(pts1_in.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
right_with_epilines, left_with_dots = draw_epilines(img2, img1, lines2, pts2_in, pts1_in)

# Save Results
cv2.imwrite("./outputs/left_with_epilines.jpg",  left_with_epilines)
cv2.imwrite("./outputs/right_with_epilines.jpg", right_with_epilines)
cv2.imwrite("./outputs/left_with_dots.jpg",      left_with_dots)
cv2.imwrite("./outputs/right_with_dots.jpg",     right_with_dots)

print("Saved to ./outputs/")