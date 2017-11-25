
import sys
import numpy as np
import scipy as sp
import cv2

# logging
import logging
FORMAT = '%(asctime)s  >>  %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
log = logging.getLogger(__name__)

python_versions = ["2.7."]
np_versions = ["1.13.3"]
sp_versions = ["0.19.1", "1.0.0"]
cv2_versions = ["2.4.9.1", "2.4.11"]


def check():

    def check_version(lib, allowed_versions):
        lib_name = lib.__name__
        lib_version = None
        if lib_name == "sys":
            lib_version = lib.version[:lib.version.index(" ")]
            log.info("{0} Version: {1} {2}".format(lib_name, lib_version, ("Pass" if any(lib_version.startswith(ver) for ver in allowed_versions) else " >> Fail <<")))
        else:
            lib_version=lib.__version__
            log.info("{0} Version: {1} {2}".format(lib_name, lib_version, (" Pass" if lib_version in allowed_versions else " >> Fail <<")))

    check_version(np, np_versions)
    check_version(sp, sp_versions)
    check_version(cv2, cv2_versions)
    check_version(sys, python_versions)

    if __name__ == "__main__":
        check()
        log.info("Hint: I am hiding something")
