import cv2
import os
import numpy as np



class ImageProcessing():
    def __init__(self):
        self.images = []
        self.cvImages = []
        for filename in os.listdir('./images'):
            self.images.append(filename)
        for image in self.images:
            self.cvImages.append(cv2.imread('./images/{}'.format(image))) #load and append image to cvImages

    def threshold(self, imageList):
        '''
        This method receives a list of images and
        returns a binarised list of the images
        '''
        self.binarisedImages = []
        #applying binarisation to all images:
        self.kernel_size = 5
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8) #simple homogeneous kernel
        for image in imageList:
            #convert to greyscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #selecting the binary inverted threshold (assigns ret to the threshold that was used and thresh to the thresholded image):
            self.ret, self.threshImage = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
            self.binarisedImages.append(self.threshImage)

        return self.binarisedImages

        
    def OuterEdgeDetection(self): #this needs to be passed an argument when called for the images to be processed
        self.contouredImages = []
        self.contourPoints = []
        #finding only the external contours using cv2.RETR_EXTERNAL argument
        for i, image in enumerate(self.cvImages):
            img_copy = image.copy()
            contours, hierarchy = cv2.findContours(self.binarisedImages[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contoured_img = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
            self.contouredImages.append(contoured_img)
            self.contourPoints.append(contours)
        #print(self.contourPoints) #debug

        #calculations to find important information of contours
        self.cropPositions = []
        for contour in self.contourPoints:
            for c in contour: #iterates through each point (x,y) of the contours array
                [x,y,w,h] = cv2.boundingRect(c)
                self.cropPositions.append([x,y,w,h])
        
        return self.contourPoints


    def cropCards(self):
        self.croppedCards = []
        for index, image in enumerate(self.cvImages):
            x,y,w,h = self.cropPositions[index]
            card = image[y:y+h, x:x+w]
            self.croppedCards.append(card)
        
        return self.croppedCards
    

    def getCentralLogo(self):
        '''
        This method returns the central logo for
        all cards.
        '''
        self.central_logo = []
        for card in (self.croppedCards):

            #getting the card height and width:
            height, width = card.shape[:2] 

            #creating new variables for cropping relative to the size of the image
            new_x = (width//3)
            new_y = (height//3)
            extra_size_x = width//12
            extra_size_y = height//12

            #create new image with just the central information in the card
            self.central_logo.append(card[(new_y - extra_size_y):(2*(new_y) + extra_size_y), (new_x - extra_size_x):(2*(new_x) + extra_size_x)])
        
        return self.central_logo
        

    def getCardColor(self):
        '''
        This method will return a list with the
        average color space for each card.
        '''

        color_spaces = []
        #THESE FOR LOOPS NEED TO OPTIMIZED LATER
        for card in self.croppedCards:
            hsv_card = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

            h,s,v = cv2.split(hsv_card)

            for h1,s1,v1 in zip(h,s,v):
                avg_h, avg_s, avg_v = (np.sum(h1))//len(h1), (np.sum(s1))//len(s1), (np.sum(v1))//len(v1)
                color_spaces.append(np.array([avg_h, avg_s, avg_v]))

        return color_spaces


    def showCards(self, imageNumber):
        '''
        this function will show the 
        operations done in all cards.
        It takes as an argument the
        number of the card that is going
        to be shown.
        '''
        #Thresholding
        cv2.imshow('Binarisation', self.binarisedImages[imageNumber])
        cv2.waitKey(0)
        cv2.destroyWindow('Binarisation')

        #outer edge detection
        cv2.imshow('Edge Detection', self.contouredImages[imageNumber])
        cv2.waitKey(0)
        cv2.destroyWindow('Edge Detection')

        #cropping the card on the outer edge
        cv2.imshow('Cropped', self.croppedCards[imageNumber])
        cv2.waitKey(0)
        cv2.destroyWindow('Cropped')

        #central logo of cards
        cv2.imshow('Central Logo', self.central_logo[imageNumber])
        cv2.waitKey(0)
        cv2.destroyWindow('Central Logo')


