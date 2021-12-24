import cv2
import numpy


'''
dumping functions that weren't used 
but can be integrated in further work
of this project.
'''

def findBiggestContour(self, contours):
    biggest = np.array([])
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:
            perimeter = cv2.arcLength(c, True)
            vertices = cv2.approxPolyDP(c, 0.02*perimeter, True)
            if area > max_area and len(vertices) == 4:
                biggest = vertices
                max_area = area

    return biggest, max_area


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


# def createOrbStream(self):
#     while self.vc.isOpened():
#         rval, frame = self.vc.read()
#         orb = cv2.ORB_create()

#         cv2.imshow('Video Frame', frame)
#         croppedCardFrame = self.cropCards(frame)


#         kp2, des2 = orb.detectAndCompute(croppedCardFrame, None)

#         imgKp2 = cv2.drawKeypoints(croppedCardFrame, kp2, None)
#         cv2.imshow('Kp1', imgKp2)
#         if cv2.waitKey(1) == 0:
#             break


def colorSeparation(self, croppedCard):
    '''
    Finding the dominant colors of
    the card using KMeans clusters.
    This will return an r,g,b tuple.
    '''
    #converting the card image to RGB
    print(type(croppedCard))
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


###This has work to be done###
def writeFeatureSpace(self):
    '''
    Dump feature space in a
    pickle file.
    '''
    dataset = ml.loadFeatureSpace(ml.ip.cvImages)
    print(len(dataset))


###This has work to be done###
def cropVideoStream(self):
    vc = cv2.VideoCapture(0)
    vc.set(10,160) #increasing brightness
    #setting width and height for image of video capture:
    heightImg = 640
    widthImg = 480

    while True:
        #creating a blank image
        blank = np.zeros((heightImg,widthImg, 3), np.uint8)

        rval, frame = vc.read()
        #frame = self.ip.threshold(frame)
        frame = self.ip.OuterEdgeDetection(frame)
        cv2.imshow('cropped card', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break



###This was used for testing inside the machine learning.py###
#ml.cropVideoStream()

#ml.findOrbMatches(ml.ip.cropCards(ml.ip.cvImages[0]), ml.ip.cropCards(ml.ip.cvImages[13]))
#print(len(ml.ip.findKeyDes()))


#print(ml.loadFeatureSpace(ml.ip.cvImages))


#ml.writeFeatureSpace()
#ml.loadTargets()
#ml.colorSeparation(ml.ip.cvImages[0])
#ml.getCardColor(ml.ip.cropCards(ml.ip.cvImages[52]))

# k = 0
# while k<56:
#     print(ml.ip.getNumberOfVertices(ml.ip.cropCards(ml.ip.cvImages[k])))
#     k += 1

# print(ml.ip.getNumberOfVertices(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getContourAxis(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getPerimeterList(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getAreaList(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getRelativeLengthList(ml.ip.cropCards(ml.ip.cvImages[4])))
# print(ml.ip.getShapeComplexityList(ml.ip.cropCards(ml.ip.cvImages[4])))


    def colorSeparation(self, croppedCard):
        '''
        Finding the dominant colors of
        the card using KMeans clusters.
        This will return an r,g,b tuple.
        '''
        #converting the card image to RGB
        #print(type(croppedCard)) #debug
        croppedCard = cv2.cvtColor(croppedCard, cv2.COLOR_BGR2HSV)

        height, width, _ = np.shape(croppedCard)

        #reshaping the image to be a simple list of rgb pixels
        image = np.float32(croppedCard.reshape(height * width, 3))

        #define the criteria for KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)

        num_clusters = 3
        attempts = 200
        
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
        #setting ranges for all possible card colors in rgb
        red_lower = (136, 87, 100)
        red_upper = (180, 255, 255)

        green_lower = (25, 52, 72)
        green_upper = (102, 255, 255)

        blue_lower = (94, 80, 2)
        blue_upper = (120, 255, 255)

        yellow_lower = (40,162,210)
        yellow_upper = (70,200,255)
        
        
        
        r,g,b = self.colorSeparation(croppedCard)
        if (r in range(red_lower[0], red_upper[0])) and (g in range(red_lower[1], red_upper[1])) and (b in range(red_lower[2], red_upper[2])):
            return 'red'
        elif (r in range(green_lower[0], green_upper[0])) and (g in range(green_lower[1], green_upper[1])) and (b in range(green_lower[2], green_upper[2])):
            return 'green'
        elif (r in range(blue_lower[0], blue_upper[0])) and (g in range(blue_lower[1], blue_upper[1])) and (b in range(blue_lower[2], blue_upper[2])):
            return 'blue'
        elif (r in range(yellow_lower[0], yellow_upper[0])) and (g in range(yellow_lower[1], yellow_upper[1])) and (b in range(yellow_lower[2], yellow_upper[2])):
            return 'yellow'
        else:
            return 'black'
        


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

