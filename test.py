# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:51:41 2021

@author: qzia
"""

import numpy as np
#import argparse
import cv2
import machine_learning
ml = machine_learning.MachineLearning()
image = cv2.imread("C:\\Users\\qzia\\Downloads\\UnoCardDetecter-master\\images\\y0.jpg")
cvGrey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
croppedCard = ml.ip.cropCards(image)
boundaries = [
	([28,57,232], [51,81,255]),
	([234, 113, 9], [255, 145, 39]),
	([40, 146, 210], [70, 200, 255]),
	([43, 10, 20], [71,255,65])
]
# loop over the boundaries
# for (lower, upper) in boundaries:
# 	# create NumPy arrays from the boundaries
# 	lower = np.array(lower, dtype = "uint8")
# 	upper = np.array(upper, dtype = "uint8")
# 	# find the colors within the specified boundaries and apply
# 	# the mask
# 	mask = cv2.inRange(croppedCard, lower, upper)
# 	output = cv2.bitwise_and(croppedCard, croppedCard, mask = mask)
# 	# show the images
# 	cv2.imshow("images", np.hstack([croppedCard, output]))
# 	cv2.waitKey(0)
    
    #################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

img = croppedCard#cv2.imread("pic/img7.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
clt = KMeans(n_clusters=2) #cluster number
clt.fit(img)

hist = find_histogram(clt)
bar = plot_colors2(hist, clt.cluster_centers_)

plt.axis("off")
plt.imshow(bar)
plt.show()
#####################################################################################





