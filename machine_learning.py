import image_processing
import numpy as np
import cv2



class MachineLearning():
    def __init__(self):
        #creating the object of the image processing class
        self.ip = image_processing.ImageProcessing()
        self.targetNames = ['0', '1', '2', '3', '4', '5', 
                        '6', '7', '8', '9', 'Reverse', 'Stop','Draw 2', 
                        'Black Blank', 'Black Swap Hands', 'Black Draw 4', 'Black Wild Card']


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

    def colorSeparation(self, croppedCard):
        '''
        Finding the dominant colors of
        the card using KMeans clusters.
        This will return an r,g,b tuple.
        '''
        #converting the card image to RGB
        #print(type(croppedCard)) #debug
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
        #setting ranges for all possible card colors in rgb
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

    def getVideoStream(self):
        vc = cv2.VideoCapture(0)
        desList = self.ip.findKeyDes()

        while True:
            rval, frame = vc.read()
            imgOriginal = frame.copy()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            
            cv2.putText(imgOriginal, '-Press esc to detect card', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(imgOriginal, '-Press space to end program', (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('frame', imgOriginal)
            
            key = cv2.waitKey(1)
            
            if (key == 27):
                cv2.imwrite('./test_images/saved.jpg', frame)
                saved = cv2.imread('./test_images/saved.jpg')

                id = self.findID(saved, desList)

                color = self.ip.findColor(imgOriginal)


                if id != -1:
                    if id not in range(13,17): #if it's not a black special card, put color text on image
                        cv2.putText(saved, color, (30,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    cv2.putText(saved, self.targetNames[id], (150,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    print(self.targetNames[id])
                else:
                    cv2.putText(saved, 'Not detected, please try again!', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

                
                cv2.imshow('classified image', saved)
                
                
            if (key == 32): #press space to end program
                cv2.destroyAllWindows()
                vc.release()
                break


    


    



