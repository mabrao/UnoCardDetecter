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
        ret, threshImage = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)

        return threshImage


    def OuterEdgeDetection(self, image):
        '''
        This method takes a cv image as an argument and
        returns a dictionary with the vertices for the image to be cropped.
        '''
        img_copy = image.copy()
        #finding only the external contours using cv2.RETR_EXTERNAL argument
        contours, hierarchy = cv2.findContours(self.threshold(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
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


    ##PREPPING FEATURE EXTRACTION FOR THE NUMBER (CENTRAL LOGO)##
    def getCentralContour(self, croppedCard):
        '''
        Returns the central contour.
        '''
        areas = []
        centralLogo = self.getCentralLogo(croppedCard)
        contours, hierarchy = cv2.findContours(self.threshold(centralLogo), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            areas.append(self.getArea(c))

        # print(areas.index(max(areas)))
        # print(max(areas))

        contoured_img = cv2.drawContours(centralLogo, contours[areas.index(max(areas))], -1, (0,255,0), 2, cv2.LINE_8)
        cv2.imshow('Central Contour', contoured_img)
        cv2.waitKey(0)
        cv2.destroyWindow('Central Contour')

        return contours[areas.index(max(areas))]

    def getPerimeter(self, contour):
        '''
        return the perimeter for a contour.
        '''
        return int(cv2.arcLength(contour, True))

    def getArea(self, contour):
        '''
        return the area for a contour
        '''
        return int(cv2.contourArea(contour))

    def getNumberOfVertices(self, croppedCard):
        '''
        returns the number of vertices for the central
        contour of the image.
        '''
        
        contour = self.getCentralContour(croppedCard)
        polygon_constant = 0.04
        perimeter = self.getPerimeter(contour)
        vertex_approx = len(cv2.approxPolyDP(contour, polygon_constant*perimeter, True))

        return vertex_approx

    def getContourAxis(self, croppedCard):
        '''
        return the major and minor axis length
        of the central contour.
        '''

        contour = self.getCentralContour(croppedCard)
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        majorAxis = (round(max(axes),2))
        minorAxis = (round(min(axes),2))

        return majorAxis, minorAxis

    ##FEATURE EXTRACTION##

    def getRelativeLength(self, croppedCard):
        '''
        return the relative length of the central contour.
        '''
        majorAxis, minorAxis = self.getContourAxis(croppedCard)
        relativeLength = round(minorAxis/majorAxis,2)

        return relativeLength

    def getShapeComplexity(self, croppedCard):
        '''
        return a list of shape complexity in percentage
        for each contour.
        '''
        contour = self.getCentralContour(croppedCard)
        perimeter = self.getPerimeter(contour)
        area = self.getArea(contour)

        return round(area/perimeter,2)


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

        #central logo of cards
        croppedCard = self.cropCards(self.cvImages[imageNumber])
        cv2.imshow('Central Logo', self.getCentralLogo(croppedCard))
        cv2.waitKey(0)
        cv2.destroyWindow('Central Logo')

