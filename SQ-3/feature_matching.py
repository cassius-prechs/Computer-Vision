import cv2
import numpy as np

orb = cv2.ORB_create(5000)

def match_features(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    )

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    )

    return pts1, pts2


def build_pair_matches(images):

    pairs = [
        ("00","01"),
        ("01","02"),

        ("10","11"),
        ("11","12"),

        ("00","10"),
        ("01","11"),
        ("02","12"),
    ]

    matches = {}

    for a, b in pairs:

        pts1, pts2 = match_features(
            images[a],
            images[b]
        )

        matches[(a,b)] = (pts1, pts2)

    return matches

def assign_track_indices(matches):
    tracks = {}  # track_id -> {image_name: (x, y)}
    next_track_id = 0
    pixel_tolerance = 1.0  

    for (a, b), (pts1, pts2) in matches.items():
        for p1, p2 in zip(pts1, pts2):
            p1_tuple = tuple(p1)
            p2_tuple = tuple(p2)
            
            found_track_id = None

            for tid, obs in tracks.items():
                if a in obs:
                    dist = np.hypot(obs[a][0] - p1_tuple[0], obs[a][1] - p1_tuple[1])
                    if dist < pixel_tolerance:
                        found_track_id = tid
                        break
                if b in obs:
                    dist = np.hypot(obs[b][0] - p2_tuple[0], obs[b][1] - p2_tuple[1])
                    if dist < pixel_tolerance:
                        found_track_id = tid
                        break

            if found_track_id is not None:
                tracks[found_track_id][a] = p1_tuple
                tracks[found_track_id][b] = p2_tuple
            else:
                tracks[next_track_id] = {
                    a: p1_tuple,
                    b: p2_tuple
                }
                next_track_id += 1

    return tracks