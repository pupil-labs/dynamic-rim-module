import logging
import os

import cv2
import numpy as np


def pick_point_in_image(rim_dir, points, npoints=4):
    """
    A function to pick the screen corners on the reference image,
    returns the corners points, and the ref image in openCV
    :param rim_dir: the directory of the reference image
    :param npoints: the number of corners to pick
    """
    # Pick the image
    image_path = os.path.join(rim_dir, "reference_image.jpeg")
    # Read the image
    image = cv2.imread(image_path)
    if points is None:
        # Create a downsample image for displaying in the corner selection. 480p height
        copy_image = image.copy()
        h, w = image.shape[0:2]
        resize_factor = 480 / h
        copy_image = cv2.resize(copy_image, (int(w * (480 / h)), 480))
        backup = copy_image.copy()
        points = []

        def pick_corners(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < npoints:
                    cv2.circle(param, (x, y), int(50 * resize_factor), (0, 0, 255), -1)
                    points.append((x, y))
                    logging.info(f"Picked point: {(x, y)}")
                    if len(points) == npoints:
                        cv2.putText(
                            param,
                            "Done, press Q to continue",
                            (int(param.shape[0] / 4), int(param.shape[1] / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            5 * resize_factor,
                            (0, 0, 255),
                            int(15 * resize_factor),
                            2,
                        )
            elif event == cv2.EVENT_FLAG_RBUTTON:
                if len(points) > 0:
                    points.pop()
                    logging.info("Removed last point")
                    cv2.addWeighted(param, 0.1, backup, 0.9, 0, param)
                    for point in points:
                        cv2.circle(
                            param, point, int(50 * resize_factor), (0, 0, 255), -1
                        )
                else:
                    logging.info("No points to remove")

        cv2.namedWindow("Pick the corners of your ROI by clicking on the image")
        cv2.setMouseCallback(
            "Pick the corners of your ROI by clicking on the image",
            pick_corners,
            copy_image,
        )
        while True:
            cv2.imshow(
                "Pick the corners of your ROI by clicking on the image", copy_image
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        # Get the points back to the original image size
        points = [
            (int(point[0] / resize_factor), int(point[1] / resize_factor))
            for point in points
        ]
    # Copy the points to the ref image
    for point in points:
        cv2.circle(image, point, 50, (0, 0, 255), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reorder the corners
    points = np.array(points)
    center = np.average(points, axis=0)
    diff_points = np.sign(points - center)
    points = points[np.lexsort((diff_points[:, 1], diff_points[:, 0]))]
    points = {
        "upper left": points[0, :],
        "lower left": points[1, :],
        "upper right": points[2, :],
        "lower right": points[3, :],
    }
    logging.info(points)
    return points, image
