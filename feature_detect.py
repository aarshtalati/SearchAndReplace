from operator import itemgetter
import collections
import numpy as np
import scipy as sp
# import scipy.signal  # one option for a 2D convolution library
import cv2
import utils


def getFeaturesFromImage(image, n_features=100):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image : numpy.ndarray
        Theimage (can be a grayscale or color image)

    n+features : int
        The number of features to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from the image
    """
    feat_detector = cv2.ORB(nfeatures=n_features)
    image_kp, image_desc = feat_detector.detectAndCompute(image, None)
    return image_kp, image_desc


def findMatchesBetweenImages(img1, img2, NUM_FEATURES, NUM_MATCHES, visualize=True):
    feat_detector = cv2.ORB(nfeatures=NUM_FEATURES)
    img1_kp, img1_desc = feat_detector.detectAndCompute(img1, None)
    img2_kp, img2_desc = feat_detector.detectAndCompute(img2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    all_matches = bfm.match(img1_desc, img2_desc)
    matches = sorted(all_matches, key=lambda x: x.distance)
    matches = matches[:NUM_MATCHES]

    # store feature locations
    img1_loc = [[], []]
    img2_loc = [[], []]
    # stitch images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    keypoints_image = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    keypoints_image[:h1, :w1, :] = img1
    keypoints_image[:h2, w1:, :] = img2
    keypoints_image[:, :, 1] = keypoints_image[:, :, 0]
    keypoints_image[:, :, 2] = keypoints_image[:, :, 0]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # loop through matches and draw lines b/w corresponding key points
    for m in matches:
        np.random.shuffle(colors)
        color = colors[0]
        y1 = int(img1_kp[m.queryIdx].pt[0])
        x1 = int(img1_kp[m.queryIdx].pt[1])
        y2 = int(img2_kp[m.trainIdx].pt[0])
        x2 = int(img2_kp[m.trainIdx].pt[1])
        cv2.line(keypoints_image, (y1, x1), (y2 + w1, x2), color, thickness=2)
        img1_loc[0].append(y1)
        img1_loc[1].append(x1)
        img2_loc[0].append(y2)
        img2_loc[1].append(x2)
    # visualize key points
    if visualize:
        file_name = "keypoints-" + utils.getTimeStamp() + ".jpg"
        cv2.imwrite(file_name, keypoints_image)
    return (img1_kp, np.array(img1_loc)), (img2_kp, np.array(img2_loc))

def findCorrespodningFeatures(matches, source_ref_matches, image_files):
    # find a feature point which matches for each album image
    counter = collections.Counter(source_ref_matches)
    # counter = collections.OrderedDict(sorted(counter.items(), key=itemgetter(1)))
    consistant_features_in_src_ref_img = [
        k for k, v in counter.iteritems() if v == len(image_files) - 1] # -1 to discount src ref img

    # find correspoding feature points in album images
    correspondance = []
    album_index = 0
    for match in matches:
        src = zip(match[1][0], match[1][1])
        album = zip(match[3][0], match[3][1])
        indices = [i for i, t in enumerate(src) if t in consistant_features_in_src_ref_img]
        for index in indices:
            x = [image_files[album_index], src[index], album[index]]
            correspondance.append(x)
        album_index += 1
    return correspondance

def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(image_1_points, image_2_points,
                                          method=cv2.RANSAC, ransacReprojThreshold=5.0)

    return homography  # mask is ignored
