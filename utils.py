import time

def getTimeStamp():
    """returns timestamp to use in file names"""
    return time.strftime("%m%d%Y_%H%M%S")