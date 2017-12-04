"""
Code adapted from CS6475 panoramas main.py
"""
import errno
import os
import logging
import datetime
import cv2
import version
import operator
import math
import numpy as np
import feature_detect as fd
import edit_detect as ed
import cluster
import triangulation
from scipy.spatial import distance

# logging
FORMAT = "%(asctime)s  >>  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
log = logging.getLogger(__name__)

NUM_FEATURES = 5000
NUM_MATCHES = 500
NUM_CLUSTERS = NUM_MATCHES / 10

SRC_FOLDER = "albums/input"
REF_FOLDER = "albums/ref"
OUT_FOLDER = "albums/output"
IMG_EXTS = set(["png", "jpeg", "jpg", "gif", "tiff", "tif", "raw", "bmp"])


def main(ref_files, image_files, output_folder):
    """main pipe line for search and replace implementation. This reads reference images from the input
    folder. It then makes identifies the edit region in each of the target images and creates new output images
    which are stored in the output folder.
    """
    src_ref_image = cv2.imread(ref_files[0])  # ref0
    edit_ref_image = cv2.imread(ref_files[1])  # ref1

    # find edits
    edits = ed.findImageDifference(src_ref_image, edit_ref_image)

    # get features from ref image

    # src_ref_kp, src_ref_desc = fd.getFeaturesFromImage(src_ref_image, NUM_FEATURES)
    # edit_ref_kp, edit_ref_desc = fd.getFeaturesFromImage(edit_ref_image, NUM_FEATURES)
    (ref_kp, ref_loc), (edit_kp, edit_loc) = fd.findMatchesBetweenImages(
        src_ref_image, edit_ref_image, NUM_FEATURES, NUM_MATCHES, visualize=False)

    matches = []
    source_ref_matches = []
    clusters = []

    for album_image in image_files:
        # iterate through album images
        #  and find matches b/w src ref img and each album img
        (src_ref_kp, src_ref_loc), (album_kp, album_loc) = fd.findMatchesBetweenImages(
            src_ref_image, cv2.imread(album_image), NUM_FEATURES, NUM_MATCHES, visualize=False)
        NUM_CLUSTERS = album_loc[0].size / 5
        clusters.append((cluster.k_means_cluster(
            album_image, album_loc, NUM_CLUSTERS, True)))

        # find edit region
        edit_top_left_x = edits[0].min()
        edit_top_left_y = edits[1].min()

        edit_bot_right_x = edits[0].max()
        edit_bot_right_y = edits[1].max()

        # find edit region center
        edit_center_y = (edits[1].max() - edits[1].min()) / 2
        edit_center_x = (edits[0].max() - edits[0].min()) / 2

        # draw rectangle
        rect = src_ref_image[:, :]
        cv2.rectangle(rect, (edit_bot_right_y, edit_bot_right_x),
                      (edit_top_left_y, edit_top_left_x), (0, 255, 0), thickness=2)
        cv2.imwrite('center.png', rect)

        # identify features which fall in edit region
        source_feature_points = zip(src_ref_loc[1], src_ref_loc[0])
        target_edit_points = []

        for i, (x, y) in enumerate(source_feature_points):
            if x > edit_top_left_x and x < edit_bot_right_x and y > edit_top_left_y and y < edit_bot_right_y:
                target_edit_points.append((x, y))

        # identify cluster centroids which fall in edit region, also calculate euclidean distance from center of edit region
        # distances = []
        # for point in (clusters[-1][1]):

        #     distances.append(distance.euclidean((edit_center_x, edit_center_y), (point[0], point[1])))

        #     p1 = np.floor(point)
        #     p2 = np.ceil(point)
        #     if p1[0] > edit_top_left_x and p1[0] < edit_bot_right_x and p1[1] > edit_top_left_y and p1[1] < edit_bot_right_y:
        #         target_edit_points.append((p1[0], p1[1]))
        #     elif p2[0] > edit_top_left_x and p2[0] < edit_bot_right_x and p2[1] > edit_top_left_y and p2[1] < edit_bot_right_y:
        #         target_edit_points.append((p2[0], p2[1]))
        # pass

        # calculate distance from matched features to the center of the edit region
        distances = {}
        matched_feature_points_in_src_ref_img = zip(
            src_ref_loc[0], src_ref_loc[1])

        temp = src_ref_image[:, :]
        for index, (x, y) in enumerate(matched_feature_points_in_src_ref_img):
            # distance between feature and edit region center in src ref img
            distances[index] = distance.euclidean(
                (edit_center_x, edit_center_y), (x, y))
            if index < 3:
                cv2.circle(temp, (x, y), 25, (255, 0, 0), thickness=2)

        cv2.imwrite('temp.png', temp)

        # get first 3 min distances for triangulation
        distances = dict(
            (sorted(distances.items(), key=operator.itemgetter(1)))[:3])

        approximation = {}
        corresponding_feature_points_in_album_img = zip(
            album_loc[0], album_loc[1])
        for k, v in distances.iteritems():
            approximation[corresponding_feature_points_in_album_img[k]] = v

        # find edit region in the target image
        p1 = np.array(list(approximation.keys()[0]))
        p2 = np.array(list(approximation.keys()[1]))
        p3 = np.array(list(approximation.keys()[2]))
        distA = approximation.values()[0]
        distB = approximation.values()[1]
        distC = approximation.values()[2]
        x = triangulation.get_location(p1, p2, p3, distA, distB, distC)

        x = np.absolute(x)
        x = np.floor(x)
        x = x.astype(np.int)

        crcl = cv2.imread(album_image)
        cv2.circle(crcl, (x[0], x[1]), 25, (255, 0, 0), thickness=2)
        cv2.imwrite('circle.png', crcl)

        # d = math.sqrt(pow((p2[0] - p1[0]), 2) + pow((p2[1] - p1[1]), 2))
        # a = ((pow(distA, 2) - pow(distB, 2) + pow(d, 2)) / (2 * d))
        # h = math.sqrt(pow(distA, 2) - pow(a, 2))
        # x2 = p1[0] + (a * (p2[0] - p1[0]) / d)
        # y2 = p1[1] + (a * (p2[1] - p1[1]) / d)
        # x3 = x2 + (h * (p2[1] - p1[1]) / d)
        # y3 = y2 - (h * (p2[0] - p1[0]) / d)

        pass

    log.info("paused")

    # album_fearure_loc : contains the location of the matching features b/w a
    # pair of source ref image and each album image

    # process edit region
    # edit_region = fd.getEditRegion( , edits)

    # process the target images
    # targets = ((name, cv2.imread(name)) for name in sorted(image_files)
    #            if path.splitext(name)[-1][1:].lower() in IMG_EXTS)

    # for each target image, apply edit to the target image
    # after finding the matching region

    # start with the first image in the folder and process each image in order
    name, target_img = targets.next()
    print "\n  Starting with: {}".format(name)
    i = 0
    for name, next_img in targets:
        if next_img is None:
            print "\nUnable to proceed: {} failed to load.".format(name)
            return

        print "  processing {}".format(name)
        matches = fd.getMatchingRegion()
        edit_xfer_img = fd.transferEdit(
            edit_region, matches, target_img, NUM_MATCHES)

    output_file_name = ("output_{0}.jpg").format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    cv2.imwrite(path.join(output_folder, output_file_name), edit_xfer_img)
    print "  Done!"


if __name__ == "__main__":

    version.check()
    """
    Read the reference and edit reference images in each subdirectory of SRC_FOLDER
    Then transfer the edits to the TGT_FOLDER images and store the output in OUT_FOLDER
    """

    subfolders = os.walk(SRC_FOLDER)
    subfolders.next()  # skip the root input folder
    for dirpath, _, fnames in subfolders:

        if fnames != []:
            image_dir = os.path.split(dirpath)[-1]
            ref_dir = os.path.join(REF_FOLDER, image_dir)
            output_dir = os.path.join(OUT_FOLDER, image_dir)

            # check whether source and edit reference files are available for input dir
            if os.path.exists(ref_dir):

                ref_files = [f for f in os.listdir(
                    ref_dir) if os.path.isfile(os.path.join(ref_dir, f))]

                if ref_files != []:
                    try:
                        os.makedirs(output_dir)
                    except OSError as exception:
                        if exception.errno != errno.EEXIST:
                            raise

                    print "Processing '" + image_dir + "' folder..."

                    ref_files = [os.path.join(
                        ref_dir, name) for name in ref_files if name.lower().endswith(tuple(IMG_EXTS))]
                    image_files = [os.path.join(
                        dirpath, name) for name in fnames if name.lower().endswith(tuple(IMG_EXTS))]

                    main(ref_files, image_files, output_dir)
                else:
                    print "image reference files not available for directory " + image_dir
