import cv2
import numpy as np


def estimate_translation(src_pts, dst_pts):
    A = np.hstack([
        np.ones((len(src_pts), 1)),
        np.zeros((len(src_pts), 1))
    ])
    diff = dst_pts - src_pts
    tx = np.linalg.lstsq(
        np.ones((len(diff), 1)),
        diff[:, 0],
        rcond=None
    )[0][0]
    ty = np.linalg.lstsq(
        np.ones((len(diff), 1)),
        diff[:, 1],
        rcond=None
    )[0][0]
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)


def estimate_similarity(src_pts, dst_pts):
    A = []
    b = []

    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, -y, 1, 0])
        A.append([y,  x, 0, 1])

        b.append(xp)
        b.append(yp)

    A = np.asarray(A)
    b = np.asarray(b)

    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, tx, ty = p

    return np.array([
        [a, -bb, tx],
        [bb,  a, ty],
        [0,   0,  1]
    ], dtype=np.float32)



def estimate_affine(src_pts, dst_pts):
    A = []
    b = []

    for (x,y),(xp,yp) in zip(src_pts,dst_pts):

        A.append([x,y,1,0,0,0])
        A.append([0,0,0,x,y,1])

        b.append(xp)
        b.append(yp)

    A = np.asarray(A)
    b = np.asarray(b)

    p,_,_,_ = np.linalg.lstsq(
        A,b,rcond=None
    )

    a00,a01,tx,a10,a11,ty = p

    return np.array([
        [a00,a01,tx],
        [a10,a11,ty],
        [0,0,1]
    ], dtype=np.float32)


def estimate_homography(src_pts, dst_pts):
    H,_ = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    return H