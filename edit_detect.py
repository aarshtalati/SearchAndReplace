import numpy as np
import cv2


def findImageDifference(image1, image2):
    """retruns edit indices
    Requirements:
    - two images must be of the same size

    Parameters:
    - reference image
    - edited image

    Returns:
    - a list of tuples containing the x and y indices of edited region
    """
    # # convert image to binary
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # _, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    # _, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    # if (image1 == image2).all():
    #     print "same images"
    #     return None  # no edits

    # image1 = np.asarray(image1, np.int32)
    # image2 = np.asarray(image2, np.int32)

    # diff = np.where(image1 == image2, 0, 255)
    # diff = diff.astype(np.uint8)

    # se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # mask = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, se1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    # mask = mask / 255
    # diff = diff * mask

    # cv2.imwrite("diff.png", diff)

    result = np.where(image1 == image2, 0, 255)
    diff_coords = np.where(result != 0)[:2]
    return diff_coords
