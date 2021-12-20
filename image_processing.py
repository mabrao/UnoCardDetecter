import cv2
import os
import numpy as np



class ImageProcessing():
    def __init__(self):
        self.cvImages = []
        self.desImages = []
        for filename in os.listdir('./images'):
            self.cvImages.append(cv2.imread('./images/{}'.format(filename))) #load and append image to cvImages
        for filename in os.listdir('./img_descriptor_train'):
            self.desImages.append(cv2.imread('./img_descriptor_train/{}'.format(filename))) #load and append image to desImages


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
    
    def noiseRemoval(self, image):
        '''
        This method receives a cv image 
        and removes noise by applying a threshold
        ans morphology.
        '''
        newImg = self.threshold(image)

        #using a matrix of size 5 as the kernel
        kernel  = np.ones((7,7), np.uint8)

        #morph close, then morph open
        newImg = cv2.morphologyEx(newImg, cv2.MORPH_CLOSE, kernel)
        newImg = cv2.morphologyEx(newImg, cv2.MORPH_OPEN , kernel)

        return newImg


    def OuterEdgeDetection(self, image):
        '''
        This method takes a cv image as an argument and
        returns a dictionary with the vertices for the image to be cropped.
        '''
        img_copy = image.copy()
        #finding only the external contours using cv2.RETR_EXTERNAL argument
        contours, hierarchy = cv2.findContours(self.threshold(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        for img in self.desImages:
            #get the cropped card from image
            croppedCard = self.cropCards(img)
            kp,des = self.createOrbCard(croppedCard)
            desList.append(des)

        return desList
    
    def createHSVTrackbars(self):
        cv2.namedWindow('HSV')
        cv2.resizeWindow('HSV', 640,240)
        cv2.createTrackbar('HUE Min', 'HSV', 0,179, self.empty)
        cv2.createTrackbar('HUE Max', 'HSV', 179,179, self.empty)
        cv2.createTrackbar('SAT Min', 'HSV', 0,255, self.empty)
        cv2.createTrackbar('SAT Max', 'HSV', 255,255, self.empty)
        cv2.createTrackbar('VALUE Min', 'HSV', 0,255, self.empty)
        cv2.createTrackbar('VALUE Max', 'HSV', 255,255, self.empty)



    def findColor(self, image):
        '''
        This method will find the most dominant color
        based on an input image, this should work with
        the camera feed as well.
        '''
        #converting the image to HSV
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # This is used for setting the bounds for each color using
        # the trackpad, this can be commented out when not calibrating:
        # h_min = cv2.getTrackbarPos('HUE Min', 'HSV')
        # h_max = cv2.getTrackbarPos('HUE Max', 'HSV')
        # s_min = cv2.getTrackbarPos('SAT Min', 'HSV')
        # s_max = cv2.getTrackbarPos('SAT Max', 'HSV')
        # v_min = cv2.getTrackbarPos('VALUE Min', 'HSV')
        # v_max = cv2.getTrackbarPos('VALUE Max', 'HSV')

        # lower_bound_trackbar = np.array([h_min,s_min,v_min])
        # upper_bound_trackbar = np.array([h_max,s_max,v_max])

        # trackbar mask (used for calibrating):
        # trackbarMask = cv2.inRange(imageHSV, lower_bound_trackbar, upper_bound_trackbar)

        # trackbar res:
        # trackbarRes = cv2.bitwise_and(imageHSV, imageHSV, mask = trackbarMask)
        # cv2.imshow('Trackbar Res', trackbarRes)

        lower_bound_red = np.array([100,190,45])
        upper_bound_red = np.array([179,255,255])
        lower_bound_yellow = np.array([0,160,100])
        upper_bound_yellow = np.array([30,255,255])
        lower_bound_blue = np.array([90,160,40])
        upper_bound_blue = np.array([150,255,255])
        lower_bound_green = np.array([40,50,0])
        upper_bound_green = np.array([90,255,255])

        #creating masks
        redMask = cv2.inRange(imageHSV, lower_bound_red, upper_bound_red)
        yellowMask = cv2.inRange(imageHSV, lower_bound_yellow, upper_bound_yellow)
        blueMask = cv2.inRange(imageHSV, lower_bound_blue, upper_bound_blue)
        greenMask = cv2.inRange(imageHSV, lower_bound_green, upper_bound_green)

        #showing the masks:
        # cv2.imshow('red mask', redMask)
        # cv2.imshow('yellow mask', yellowMask)
        # cv2.imshow('blue mask', blueMask)
        # cv2.imshow('green mask', greenMask)

        #creating results with bitwise ands
        resRed = cv2.bitwise_and(imageHSV, imageHSV, mask = redMask)
        resYellow = cv2.bitwise_and(imageHSV, imageHSV, mask = yellowMask)
        resBlue = cv2.bitwise_and(imageHSV, imageHSV, mask = blueMask)
        resGreen = cv2.bitwise_and(imageHSV, imageHSV, mask = greenMask)

        # cv2.imshow('red res', resRed)
        # cv2.imshow('yellow res', resYellow)
        # cv2.imshow('blue res', resBlue)
        # cv2.imshow('green res', resGreen)

        #convert masks to 3 channels:
        # redMask = cv2.cvtColor(redMask, cv2.COLOR_GRAY2BGR)
        # yellowMask = cv2.cvtColor(yellowMask, cv2.COLOR_GRAY2BGR)
        # blueMask = cv2.cvtColor(blueMask, cv2.COLOR_GRAY2BGR)
        # greenMask = cv2.cvtColor(greenMask, cv2.COLOR_GRAY2BGR)

        #Find the biggest external contour of each mask:
        biggestContours = {}
        
        contoursRed, hierarchy = cv2.findContours(self.noiseRemoval(resRed), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursYellow, hierarchy = cv2.findContours(self.noiseRemoval(resYellow), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursBlue, hierarchy = cv2.findContours(self.noiseRemoval(resBlue), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursGreen, hierarchy = cv2.findContours(self.noiseRemoval(resGreen), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contoursRed) != 0:
            maxContourRed = cv2.contourArea(max(contoursRed, key = cv2.contourArea))
            biggestContours['Red'] = maxContourRed
        
        if len(contoursYellow) != 0:
            maxContourYellow = cv2.contourArea(max(contoursYellow, key = cv2.contourArea))
            biggestContours['Yellow'] = maxContourYellow

        if len(contoursBlue) != 0:
            maxContourBlue = cv2.contourArea(max(contoursBlue, key = cv2.contourArea))
            biggestContours['Blue'] = maxContourBlue

        if len(contoursGreen) != 0:
            maxContourGreen = cv2.contourArea(max(contoursGreen, key = cv2.contourArea))
            biggestContours['Green'] = maxContourGreen


        ###DEBUGGING###

        if len(biggestContours) != 0:
            #print(max(biggestContours))
            return max(biggestContours)
        else:
            return 'Failed'

        # img_copy1 = image.copy()
        # img_copy2 = image.copy()
        # img_copy3 = image.copy()
        # img_copy4 = image.copy()

        # contoured_red = cv2.drawContours(img_copy1, contoursRed, -1, (0, 255, 0), 2)
        # contoured_yellow = cv2.drawContours(img_copy2, contoursYellow, -1, (0, 255, 0), 2)
        # contoured_blue = cv2.drawContours(img_copy3, contoursBlue, -1, (0, 255, 0), 2)
        # contoured_green = cv2.drawContours(img_copy3, contoursGreen, -1, (0, 255, 0), 2)
        # cv2.imshow('red contour', contoured_red)
        # cv2.imshow('yellow contour', contoured_yellow)
        # cv2.imshow('blue contour', contoured_blue)
        # cv2.imshow('green contour', contoured_green) 

        #stacking the masks together to display later
        # hstack = np.hstack([image,redMask,yellowMask,blueMask, greenMask])
        # cv2.imshow('HStack', hstack)
        # result = cv2.bitwise_and(image, image, mask = mask)
        # cv2.imshow('HSV Color Space', imageHSV)
        # cv2.imshow('Result', result)
        # cv2.imshow('trackbar mask ', trackbarMask)
        
    
    def empty(self, a):
        pass



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
