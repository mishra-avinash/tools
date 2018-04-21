# Source: https://github.com/nicolefinnie/kaggle-dsb2018/blob/master/src/modules/image_processing.py


import numpy as np
import cv2


def mark_contours(mask):
    """Pass in a 2-dimensional grayscale mask, mark contours of the mask,
    and return the 2-dimensional grayscale contoured mask
    """
    padded = 2
    mask = mask.astype(np.uint8) * 255

    background = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    padded_background = np.pad(background.copy(), ((padded, padded), (padded, padded)), 'edge')
    background_rgb = cv2.cvtColor(padded_background, cv2.COLOR_GRAY2RGB)

    padded_mask = np.pad(mask.copy(), ((padded, padded), (padded, padded), (0, 0)), 'edge')

    _, thresh = cv2.threshold(padded_mask, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contoured_rgb = cv2.drawContours(background_rgb, contours, -1, (255, 255, 255), 1)
    contoured_gray = cv2.cvtColor(contoured_rgb, cv2.COLOR_RGB2GRAY)
    # workaround due to OpenCV issue, the contour starts from the 1st pixel
    contoured_mask = contoured_gray[padded:-padded, padded:-padded]

    return contoured_mask


def mark_mask_on_image(mask, image, color=(255, 0, 0)):
    """ Mark red contours of the given masks on the given image
    """
    mask = mask.astype(np.uint8) * 255
    image_color = image.copy()
    if image.ndim is 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return cv2.drawContours(image_color, contours, -1, color, 1)
