import cv2
import os
import numpy as np



class ImageProcessing():
    def __init__(self):
        self.cvImages = []
        for filename in os.listdir('./images'):
            self.cvImages.append(cv2.imread('./images/{}'.format(filename))) #load and append image to cvImages

    def threshold(self, image):
        '''
        This method receives a cv image as
        an argument and returns a thresholded image.
        '''
        #convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #selecting the binary inverted threshold (assigns ret to the threshold that was used and thresh to the thresholded image):
        ret, threshImage = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)

        return threshImage
    

    def OuterEdgeDetection(self, image):
        '''
        This method takes a cv image as an argument and
        returns a dictionary with the vertices for the image to be cropped.
        '''
        img_copy = image.copy()
        #finding only the external contours using cv2.RETR_EXTERNAL argument
        contours, hierarchy = cv2.findContours(self.threshold(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        #making this a self variable so it can be accessed on the show cards method:
        self.contoured_img = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2) 

        #calculations to find where to crop the card
        self.cropPositions = []
        for c in contours: #iterates through each point (x,y) of the contour array
            [x,y,w,h] = cv2.boundingRect(c)
            cropPositions = {'x':x,'y':y,'w':w,'h':h}

        return cropPositions

    def cropCards(self, image):
        '''
        This method takes a cv image as an argument
        and returns a cropped image with only the card,
        '''
        cropPositions = self.OuterEdgeDetection(image)
        x,y,w,h = cropPositions['x'], cropPositions['y'], cropPositions['w'], cropPositions['h']
        croppedCard = image[y:y+h, x:x+w]

        return croppedCard


    ##PREPPING FEATURE EXTRACTION FOR THE NUMBER (CENTRAL LOGO) WITH ORB##
    def createOrbCard(self, croppedCard):
        '''
        This function will use the Oriented Fast and Rotated
        algorithm. These algorithm takes a cropped card as
        an input (open cv image) and returns the keypoints (tuple)
        and the description of those keypoints (numpy.ndarray)
        '''
        croppedCard = cv2.cvtColor(croppedCard, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)

        kp, des = orb.detectAndCompute(croppedCard, None)

        ##DEBUG AND VISUALIZATION

        # imgKp1 = cv2.drawKeypoints(croppedCard,kp,None)
        # #print(des1.shape)

        # cv2.imshow('Kp1', imgKp1)
        # cv2.waitKey(0)
        # cv2.destroyWindow('Kp1')
        # print(type(kp), type(des))

        return kp, des
    
    def findKeyDes(self):
        desList = []
        for img in self.cvImages:
            #get the cropped card from image
            croppedCard = self.cropCards(img)
            kp,des = self.createOrbCard(croppedCard)
            desList.append(des)

        return desList



    ##SHOW OPERATIONS DONE ON THE CARDS##
    def showCards(self, imageNumber):
        '''
        this function will show the
        operations done in all cards.
        It takes as an argument the
        number of the card that is going
        to be shown.
        '''
        #Thresholding
        cv2.imshow('Binarisation', self.threshold(self.cvImages[imageNumber]))
        cv2.waitKey(0)
        cv2.destroyWindow('Binarisation')

        #outer edge detection
        self.OuterEdgeDetection(self.cvImages[imageNumber]) #perform edge detection on the selected image and store the contour image
        cv2.imshow('Edge Detection', self.contoured_img)
        cv2.waitKey(0)
        cv2.destroyWindow('Edge Detection')

        #cropping the card on the outer edge
        cv2.imshow('Cropped', self.cropCards(self.cvImages[imageNumber]))
        cv2.waitKey(0)
        cv2.destroyWindow('Cropped')
