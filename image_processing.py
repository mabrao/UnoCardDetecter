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
        contours, hierarchy = cv2.findContours(self.threshold(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contoured_img = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2) #making this a self variable so it can be accessed on the show cards method

        #calculations to find where to crop the card
        self.cropPositions = []
        for c in contours: #iterates through each point (x,y) of the contour array
            [x,y,w,h] = cv2.boundingRect(c)
            cropPositions = {'x':x,'y':y,'w':w,'h':h}
        
        return cropPositions

    def cropCards(self, image):
        '''
        This method takes a cv image as an argument
        and returns a cropped image with only the card.
        '''
        cropPositions = self.OuterEdgeDetection(image)
        x,y,w,h = cropPositions['x'], cropPositions['y'], cropPositions['w'], cropPositions['h']
        croppedCard = image[y:y+h, x:x+w]
        
        return croppedCard
    

    def getCentralLogo(self, croppedCard):
        '''
        This method takes a cv cropped card as an argument
        and returns the cropped central logo of the image.
        '''
        #getting the card height and width:
        height, width = croppedCard.shape[:2] 

        #creating new variables for cropping relative to the size of the image
        new_x = (width//3)
        new_y = (height//3)
        extra_size_x = width//12
        extra_size_y = height//12

        #create new image with just the central information in the card
        central_logo = (croppedCard[(new_y - extra_size_y):(2*(new_y) + extra_size_y), (new_x - extra_size_x):(2*(new_x) + extra_size_x)])
        
        return central_logo
        

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

        #central logo of cards
        croppedCard = self.cropCards(self.cvImages[imageNumber])
        cv2.imshow('Central Logo', self.getCentralLogo(croppedCard))
        cv2.waitKey(0)
        cv2.destroyWindow('Central Logo')


