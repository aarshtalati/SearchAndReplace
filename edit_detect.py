import numpy as np
import cv2


def findImageDifference(image1, image2, save_to_file=None):
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # _, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    # _, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
    
    # image1 = np.asarray(image1, np.int32)
    # image2 = np.asarray(image2, np.int32)

    # diff = np.subtract(image1, image2)

    diff = cv2.subtract(image1, image2)
    
    if save_to_file is not None:
        cv2.imwrite(save_to_file, diff)

    diff = np.asarray(diff, np.int32)
    x = np.where(diff!=0)
    return x