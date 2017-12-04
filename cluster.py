import numpy as np
import cv2
from matplotlib import pyplot as plt

def k_means_cluster(img, features, K=10, visualize = False, initial_means=None):

    img_rgb = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
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
        coords = []

        # show image
        fig = plt.figure()
        plt.imshow(img_rgb)

        # Plot the cluster data
        ax = fig.add_subplot(111)

        for i in range(max(label)+1):
            cluster_points = features[label.ravel()==i]
            x,y = cluster_points[:,0],cluster_points[:,1]
            scatter = ax.scatter(x, y, c="k", s=3)

        ax.scatter(center[:, 0], center[:, 1], s=20, c='y', marker='s')
        plt.xlabel('X'),plt.ylabel('Y')
        # plt.show()

        file_name = "cluster_" + img.split("/")[-1]
        fig = plt.gcf()
        fig.savefig(file_name)

    return label, center

