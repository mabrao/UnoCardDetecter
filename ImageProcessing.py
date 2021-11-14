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

    def threshold(self):
        self.binarisedImages = []
        self.dilatedImages = []
        self.gradientImages = []
        #applying binarisation to all images:
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8) #simple homogeneous kernel
        for image in self.cvImagesGrey:
            #selecting the binary inverted threshold (assigns ret to the threshold that was used and thresh to the thresholded image):
            self.ret, self.threshImage = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)
            self.gradient = cv2.morphologyEx(self.threshImage, cv2.MORPH_GRADIENT, kernel)
            self.binarisedImages.append(self.threshImage) 
            self.gradientImages.append(self.gradient)

        print(self.binarisedImages)
        cv2.imshow('Binarisation', self.binarisedImages[-1])
        cv2.waitKey(0)
        cv2.destroyWindow('Binarisation')
        # cv2.imshow('Gradient', self.gradientImages[-1])
        # cv2.waitKey(0)
        # cv2.destroyWindow('Gradient')
    
    def edgeDetection(self):
        self.cannyImages = []
        for Binarisedimage in self.binarisedImages:
            cannyImage = cv2.Canny(Binarisedimage, 50, 100)
            contours, hierarchy = cv2.findContours(cannyImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(cannyImage, contours, -1, (0, 255, 0), 1)
            self.cannyImages.append(cannyImage)
            
        cv2.imshow('Edge Detection', self.cannyImages[-1])
        cv2.waitKey(0)
        cv2.destroyWindow('Edge Detection')

        



ip = ImageProcessing()
ip.threshold()
ip.edgeDetection()

