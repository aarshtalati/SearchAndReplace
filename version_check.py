import sys
import numpy as np
import scipy as sp
import cv2

class version():
    def __init__(self):
        self.python_versions = ["2.7."]
        self.np_versions = ["1.13.3"]
        self.sp_versions = ["0.19.1", "1.0.0"]
        self.cv2_versions = ["2.4.9.1", "2.4.11"]

    def check_version(self, lib, allowed_versions):
        lib_name = lib.__name__
        lib_version = None
        if lib_name == "sys":
            lib_version = lib.version[:lib.version.index(' ')]
            print lib_name, "Version:", lib_version, " Pass" if any(lib_version.startswith(ver) for ver in allowed_versions) else " >> Fail <<"
        else:
            lib_version = lib.__version__
            print lib_name, "Version:", lib_version, " Pass" if lib_version in allowed_versions else " >> Fail <<"

    def check(self):
        self.check_version(np, self.np_versions)
        self.check_version(sp, self.sp_versions)
        self.check_version(cv2,self.cv2_versions)
        self.check_version(sys,self.python_versions)

    if __name__ == "__main__":
        print "Hint: I am hiding something"

