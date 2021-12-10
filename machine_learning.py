import image_processing
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os


class MachineLearning():
    def __init__(self):
        #creating the object of the image processing class
        self.ip = image_processing.ImageProcessing()

    
    def colorSeparation(self, croppedCard):
        '''
        Finding the dominant colors of
        the card using KMeans clusters.
        This will return an r,g,b tuple.
        '''
        #converting the card image to RGB
        croppedCard = cv2.cvtColor(croppedCard, cv2.COLOR_BGR2RGB)

        height, width, _ = np.shape(croppedCard)

        #reshaping the image to be a simple list of rgb pixels
        image = np.float32(croppedCard.reshape(height * width, 3))

        #define the criteria for KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

        num_clusters = 3
        attempts = 100
        
        ret, label, center = cv2.kmeans(image,num_clusters,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

        #converting the center of data back into uint8
        center = np.uint8(center)
        for r,g,b in center:
            if (r < 60) and (g < 60) and (b < 60): ##THIS CAN BE OPTIMIZED TO 2 LINES
                # print(f'{r = } {g = } {b = }')
                # print('black')
                pass
            elif (r > 190) and (g > 190) and (b > 190):
                # print(f'{r = } {g = } {b = }')
                # print('white')
                pass
            else:
                #print(f'{r = } {g = } {b = }')
                return (r,g,b)
    
    def getCardColor(self, croppedCard):
        '''
        This method will be returning a string
        with the color of the card in 
        all lower case letters.
        '''
        #setting ranges for all possible card colors in hsv
        red_lower = (232,57,28)
        red_upper = (255,81,52)
    
        green_lower = (20,51,43)
        green_upper = (65,255,71)
    
        blue_lower = (9,113,234)
        blue_upper = (39,145,255)

        yellow_lower = (210,162,40)
        yellow_upper = (255,200,70)
        
        r,g,b = self.colorSeparation(croppedCard)

        #next section code which is commented out can be used for putting the text on the cards:
        # textColor = (255,255,255)
        # fontScale = 1
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # org = (50,50)

        if (r in range(red_lower[0], red_upper[0])) and (g in range(red_lower[1], red_upper[1])) and (b in range(red_lower[2], red_upper[2])):
            # cv2.putText(croppedCard, 'RED', org, font, fontScale, textColor, 2, cv2.LINE_AA)
            # cv2.imshow('color finder', croppedCard)
            # cv2.waitKey(0)
            # cv2.destroyWindow('color finder')
            return 'red'
        elif (r in range(green_lower[0], green_upper[0])) and (g in range(green_lower[1], green_upper[1])) and (b in range(green_lower[2], green_upper[2])):
            # cv2.putText(croppedCard, 'GREEN', org, font, fontScale, textColor, 2, cv2.LINE_AA)
            # cv2.imshow('color finder', croppedCard)
            # cv2.waitKey(0)
            # cv2.destroyWindow('color finder')
            return 'green'
        elif (r in range(blue_lower[0], blue_upper[0])) and (g in range(blue_lower[1], blue_upper[1])) and (b in range(blue_lower[2], blue_upper[2])):
            # cv2.putText(croppedCard, 'BLUE', org, font, fontScale, textColor, 2, cv2.LINE_AA)
            # cv2.imshow('color finder', croppedCard)
            # cv2.waitKey(0)
            # cv2.destroyWindow('color finder')
            return 'blue'
        elif (r in range(yellow_lower[0], yellow_upper[0])) and (g in range(yellow_lower[1], yellow_upper[1])) and (b in range(yellow_lower[2], yellow_upper[2])):
            # cv2.putText(croppedCard, 'YELLOW', org, font, fontScale, textColor, 2, cv2.LINE_AA)
            # cv2.imshow('color finder', croppedCard)
            # cv2.waitKey(0)
            # cv2.destroyWindow('color finder')
            return 'yellow'
        else:
            # cv2.putText(croppedCard, 'BLACK', org, font, fontScale, textColor, 2, cv2.LINE_AA)
            # cv2.imshow('color finder', croppedCard)
            # cv2.waitKey(0)
            # cv2.destroyWindow('color finder')
            return 'black'

    #create a method to gather all features and store them in a dictionary.
    def createFeatureSpace(self, croppedCard):
        '''
        This method will create a dictionary with 
        all features for a card.
        '''
        featureSpace = {}
        try:
            featureSpace['color'] = self.getCardColor(croppedCard)
        except Exception as e:
            print(e)
            featureSpace['color'] = None
        try:    
            featureSpace['corners'] = self.ip.getNumberOfVertices(croppedCard)
        except:
            featureSpace['corners'] = None
        try:
            featureSpace['relative length'] = self.ip.getRelativeLength(croppedCard)
        except Exception as e:
            print(e)
            featureSpace['relative length'] = None
        try:
            featureSpace['shape complexity'] = self.ip.getShapeComplexity(croppedCard)
        except:
            featureSpace['shape complexity'] = None

        return featureSpace
    
    
    #create a method to iterate through all files and store a list of the dictionaries with all features
    #this will be the data to train and test the data set
    def loadFeatureSpace(self, trainData):
        '''
        This method will return a list of dictionaries
        that will be used to train the data.
        '''
        features = []
        for card in trainData:
            features.append(self.createFeatureSpace(card))
            
        return features
    
    def loadTargets(self):
        targets = [0,1,2,3,4,5,6,7,8,9,10,11,12,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        0,1,2,3,4,5,6,7,8,9,10,11,12,
        0,1,2,3,4,5,6,7,8,9,10,11,12]

        return targets
        


    
    def writeFeatureSpace(self):
        '''
        Dump feature space in a
        pickle file
        '''
        dataset = ml.loadFeatureSpace(ml.ip.cvImages)
        print(len(dataset))

    
        



ml = MachineLearning()

#print(ml.loadFeatureSpace(ml.ip.cvImages))


#ml.writeFeatureSpace()
#ml.loadTargets()
#ml.colorSeparation(ml.ip.cvImages[0])
#ml.getCardColor(ml.ip.cropCards(ml.ip.cvImages[52]))

k = 0
while k<56:
    print(ml.ip.getNumberOfVertices(ml.ip.cropCards(ml.ip.cvImages[k])))
    k += 1

# print(ml.ip.getNumberOfVertices(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getContourAxis(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getPerimeterList(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getAreaList(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getRelativeLengthList(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getShapeComplexityList(ml.ip.cropCards(ml.ip.cvImages[4])))

