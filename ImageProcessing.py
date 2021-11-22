import cv2
import os
import numpy as np



class ImageProcessing():
    def __init__(self):
        self.images = []
        self.cvImages = []
        self.cvImagesGrey = [] #IS THIS NECESSARY OR CAN I JUST CONVERT TO GREY SCALE LATER?
        for filename in os.listdir('./images'):
            self.images.append(filename)
        for image in self.images:
            self.cvImages.append(cv2.imread('./images/{}'.format(image))) #load and append image to cvImages
            self.cvImagesGrey.append(cv2.imread('./images/{}'.format(image), 0)) #load and append image in grey scale to cvImages

    def threshold(self): #this needs to be passed an argument when called for the images to be processed
        self.binarisedImages = []
        self.gaussianImages = []
        #applying binarisation to all images:
        self.kernel_size = 5
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8) #simple homogeneous kernel
        for image in self.cvImagesGrey:
            #selecting the binary inverted threshold (assigns ret to the threshold that was used and thresh to the thresholded image):
            self.ret, self.threshImage = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
            self.binarisedImages.append(self.threshImage)
            #applying the gaussian blur for better image detection:
            self.gaussianImages.append(cv2.GaussianBlur(image, (3,3), 0))

        #TESTING PURPOSES
        cv2.imshow('Binarisation', self.binarisedImages[3])
        cv2.waitKey(0)
        cv2.destroyWindow('Binarisation')

    def edgeDetection(self): #this needs to be passed an argument when called for the images to be processed
        self.contouredImages = []
        self.contourPoints = []
        #finding only the external contours using cv2.RETR_EXTERNAL argument
        for i, image in enumerate(self.cvImages):
            img_copy = image.copy()
            contours, hierarchy = cv2.findContours(self.binarisedImages[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contoured_img = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
            self.contouredImages.append(contoured_img)
            self.contourPoints.append(contours)
        print(self.contourPoints)
        #calculations to find important information of contours
        self.cropPositions = []
        for contour in self.contourPoints:
            for c in contour: #iterates through each point (x,y) of the contours array
                #area = cv2.contourArea(c)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                [x,y,w,h] = cv2.boundingRect(c)
                self.cropPositions.append([x,y,w,h])

        #TESTING PURPOSES
        cv2.imshow('Edge Detection', self.contouredImages[3])
        cv2.waitKey(0)
        cv2.destroyWindow('Edge Detection')
    
    def cropCards(self):
        self.cropped = []
        for index, image in enumerate(self.cvImages):
            x,y,w,h = self.cropPositions[index]
            card = image[y:y+h, x:x+w]
            self.cropped.append(card)
        
        #TESTING PURPOSES
        cv2.imshow('Cropped', self.cropped[3])
        cv2.waitKey(0)
        cv2.destroyWindow('Cropped')
        



ip = ImageProcessing()
ip.threshold()
ip.edgeDetection()
ip.cropCards()
