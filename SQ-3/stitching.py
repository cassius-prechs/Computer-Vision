import cv2
import numpy as np


def get_corners(img):
    h, w = img.shape[:2]

    return np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)


def compute_canvas(images, transforms, model):
    all_pts = []

    for name, img in images.items():

        corners = get_corners(img)

        if model == "homography":

            warped = cv2.perspectiveTransform(
                corners,
                transforms[name]
            )

        else:

            warped = cv2.transform(
                corners,
                transforms[name][:2]
            )

        all_pts.append(
            warped.reshape(-1, 2)
        )

    all_pts = np.vstack(all_pts)

    xmin = np.floor(
        np.min(all_pts[:, 0])
    )

    ymin = np.floor(
        np.min(all_pts[:, 1])
    )

    xmax = np.ceil(
        np.max(all_pts[:, 0])
    )

    ymax = np.ceil(
        np.max(all_pts[:, 1])
    )

    width = int(xmax - xmin)
    height = int(ymax - ymin)

    return width, height, xmin, ymin


def stitch_images(
    images,
    transforms,
    model
):

    canvas_w, canvas_h, xmin, ymin = compute_canvas(
        images,
        transforms,
        model
    )

    canvas = np.zeros(
        (canvas_h, canvas_w, 3),
        dtype=np.float32
    )

    count = np.zeros(
        (canvas_h, canvas_w),
        dtype=np.float32
    )

    for name, img in images.items():

        T = transforms[name]
        h, w = img.shape[:2]
        
        img_mask = np.ones((h, w), dtype=np.float32)

        if model == "homography":

            offset = np.array([
                [1, 0, -xmin],
                [0, 1, -ymin],
                [0, 0, 1]
            ], dtype=np.float32)

            T2 = offset @ T

            warped = cv2.warpPerspective(
                img.astype(np.float32),
                T2,
                (canvas_w, canvas_h)
            )
            
            warped_mask = cv2.warpPerspective(
                img_mask,
                T2,
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST # マスクは補間しない
            )

        else:

            T2 = T.copy()

            T2[0, 2] -= xmin
            T2[1, 2] -= ymin

            warped = cv2.warpAffine(
                img.astype(np.float32),
                T2[:2],
                (canvas_w, canvas_h)
            )
            
            warped_mask = cv2.warpAffine(
                img_mask,
                T2[:2],
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST
            )

        canvas += warped
        count += warped_mask

    count = np.maximum(count, 1)

    result = (
        canvas /
        count[:, :, None]
    )

    return np.clip(result, 0, 255).astype(np.uint8)
