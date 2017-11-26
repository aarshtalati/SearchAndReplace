import numpy as np
import cv2

"""retruns edit indices

Requirements:
- two images must be of the same size

Parameters:
- reference image
- edited image

Returns:
- a list of tuples containing the x and y indices of edited region

"""


def findImageDifference(image1, image2):

    # convert image to binary
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    if (image1 == image2).all():
        print "same images"
        return None  # no edits

    image1 = np.asarray(image1, np.int32)
    image2 = np.asarray(image2, np.int32)

    indices = np.where(image1 == image2)
    return indices
