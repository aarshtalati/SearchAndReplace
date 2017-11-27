"""
Code adapted from CS6475 panoramas main.py
"""
import errno
import os
import logging
import datetime
import cv2
import version
import feature_detect as fd
import edit_detect as ed

# logging
FORMAT = "%(asctime)s  >>  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
log = logging.getLogger(__name__)

NUM_FEATURES = 1000
NUM_MATCHES = 100
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
    src_ref_kp, edit_ref_kp, matches = fd.findMatchesBetweenImages(src_ref_image, edit_ref_image, NUM_FEATURES, NUM_MATCHES)


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
