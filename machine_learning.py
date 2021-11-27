import image_processing
import numpy as np


class MachineLearning():
    def __init__(self):
        #creating the object of the image processing class
        self.ip = image_processing.ImageProcessing()
        self.ip.threshold(self.ip.cvImages)
        self.ip.OuterEdgeDetection()
        self.ip.cropCards()
        self.ip.getCentralLogo()
        self.color_spaces = self.ip.getCardColor()
    
    def colorSeparation(self):
        #setting the classes for color detection
        low_black_bound, high_black_bound = [], []
        low_red_bound, high_red_bound = [0,100,58], [0,100,100]
        low_blue_bound, high_blue_bound = [240,100,39], [240,100,100]
        low_green_bound, high_green_bound = [120,100,19], [120,100,100]
        low_yellow_bound, high_yellow_bound = [60,100,58], [60,100,100]
        for h,s,v in self.color_spaces:
            #convert to 100% scale
            h,s,v = (int((h/255.0)*100.0)), (int((s/255.0)*100.0)), (int((v/255.0)*100.0))
            
            # if h < 


ml = MachineLearning()
#image_processing.ImageProcessing.showCards(ml.ip, 15)
ml.colorSeparation()




