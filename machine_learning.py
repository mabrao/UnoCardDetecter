import image_processing_2
import numpy as np
from sklearn.cluster import KMeans
import cv2


class MachineLearning():
    def __init__(self):
        #creating the object of the image processing class
        self.ip = image_processing_2.ImageProcessing()
        self.ip.getCentralLogo(self.ip.cropCards(self.ip.cvImages[0]))
        self.ip.showCards(1)        
    
    def colorSeparation(self):
        '''
        Finding the dominant colors of
        the card using KMeans clusters.
        '''
        pass

    
        



ml = MachineLearning()
#image_processing.ImageProcessing.showCards(ml.ip, 15)
ml.colorSeparation()