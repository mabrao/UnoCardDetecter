import image_processing
import numpy as np
import cv2



class MachineLearning():
    def __init__(self):
        #creating the object of the image processing class
        self.ip = image_processing.ImageProcessing()
        self.targetNames = ['Blue 0', 'Blue 1', 'Blue 2', 'Blue 3', 'Blue 4', 'Blue 5', 
                        'Blue 6', 'Blue 7', 'Blue 8', 'Blue 9', 'Blue Reverse', 'Blue Stop', 
                        'Blue Draw 2', 'Green 0', 'Green 1', 'Green 2', 'Green 3', 'Green 4', 
                        'Green 5', 'Green 6', 'Green 7', 'Green 8', 'Green 9', 'Green Reverse', 
                        'Green Stop', 'Green Draw 2', 'Black Blank', 'Black Swap Hands', 
                        'Black Draw 4', 'Black Wild Card', 
                        'Red 0', 'Red 1', 'Red 2', 'Red 3', 'Red 4', 'Red 5', 'Red 6', 
                        'Red 7', 'Red 8', 'Red 9', 'Red Reverse', 'Red Stop', 'Red Draw 2', 
                        'Yellow 0', 'Yellow 1', 'Yellow 2', 'Yellow 3', 'Yellow 4', 'Yellow 5', 
                        'Yellow 6', 'Yellow 7', 'Yellow 8', 'Yellow 9', 'Yellow Reverse', 'Yellow Stop', 
                        'Yellow Draw 2']


    def findID(self, videoCapture, desList, threshold = 7):
        #kp2,des2 = self.ip.createOrbCard(videoCapture)
        orb = cv2.ORB_create(nfeatures=1000)
        kp2,des2 = orb.detectAndCompute(videoCapture, None)
        bf = cv2.BFMatcher()
        desList = self.ip.findKeyDes()
        matchList = []
        finalValue = -1
        try:
            for des in desList:
                matches = bf.knnMatch(des,des2, k=2)
                goodMatches = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        goodMatches.append([m])
                matchList.append(len(goodMatches))
        except:
            pass
        #print(matchList) #number of good matches found with each of the 'train images'
        if len(matchList) != 0:
            if max(matchList) > threshold:
                finalValue = matchList.index(max(matchList))
        
        return finalValue


    def getVideoStream(self):
        vc = cv2.VideoCapture(0)
        desList = self.ip.findKeyDes()

        while True:
            rval, frame = vc.read()
            imgOriginal = frame.copy()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            id = self.findID(frame, desList)
            # print(id)
            if id != -1:
                cv2.putText(imgOriginal, self.targetNames[id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow('frame', imgOriginal)
            key = cv2.waitKey(1)
            
            if (key == 27):
                break
    
    
    #2 things left to do:

    #1) find the card color with a mask and cv2.inRange

    #2) create a function to input a card into the find ID file instead of the video stream


    



