import numpy as np
import cv2
from matplotlib import pyplot as plt

def k_means_cluster(features, K=10, visualize = False, initial_means=None):

    features = np.float32(features)
    features = np.transpose(features)

    # define criteria and apply kmeans()
    # episilon = 1.0 and iteration = 20
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    if cv2.__version__.split('.')[0] == '3':
        ret,label,center=cv2.kmeans(features, K, None, criteria, 20,cv2.KMEANS_RANDOM_CENTERS)
    elif cv2.__version__.split('.')[0] == '2':
        ret,label,center=cv2.kmeans(features, K, criteria, 20,cv2.KMEANS_RANDOM_CENTERS)
    

    if (visualize):
        # Now separate the data, Note the flatten()
        A = features[label.ravel()==0]
        B = features[label.ravel()==1]

        # Plot the data
        plt.scatter(A[:,0],A[:,1])
        plt.scatter(B[:,0],B[:,1],c = 'r')
        plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
        plt.xlabel('Height'),plt.ylabel('Weight')
        plt.show()

    return label, center

