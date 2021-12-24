'''
Uno Card Detecter by Matheus Abrao.
Run this file to start the program.
'''

import machine_learning
import tkinter as tk
from tkinter import filedialog as fd
import cv2


class Gui():
    def __init__(self):
        self.window = tk.Tk()
        self.ml = machine_learning.MachineLearning()
        
    
    def configWindow(self):
        self.window.title(' Uno Card Detection ')
        self.window.geometry('350x200')


        videoButton = tk.Button(self.window, text = 'Detect with camera', fg='black', command = self.ml.getVideoStream, height=5, width=20)
        selectFileButton = tk.Button(self.window, text= 'Browse Files', command = self.selectFiles, height=5, width=20)

        # videoButton.grid(column=5, row=5)
        # selectFileButton.grid(column=10, row=10)
        videoButton.pack()
        selectFileButton.pack()
    
    def selectFiles(self):
        filetypes = (('jpg file', '*.jpg'), ('All files', '*.*'))

        filename = fd.askopenfilename(title= 'Browse files', initialdir= './', filetypes= filetypes)

        #print(filename)
        cvImage = cv2.imread(filename)
        cvGrey = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)

        #resize image: - used for better visualization of images that have different sizes
        #print(f'width = {cvImage.shape[1]} height = {cvImage.shape[0]}')
        #dimensions = (640,480)
        #cvImage = cv2.resize(cvImage, dimensions, interpolation=cv2.INTER_AREA)

        ##Color detection using kmeans clustering##
        #crop card:
        croppedCard = self.ml.ip.cropCards(cvImage)
        #print(croppedCard.shape) #debug
        #get color
        color_bar, _ = self.ml.getCardColorNew(croppedCard)
        #find color name of cropped card:
        color_name = self.ml.colorNameDetection(croppedCard)

        ## apply ORB on image file
        desList = self.ml.ip.findKeyDes() #get description list of each train image
        id = self.ml.findID(cvGrey, desList) #get id of grey input image
        
        

        if id != -1: #checking if there was any detection
            if id not in range(13,17): #if it's not a black special card, put color text on image
                #cv2.putText(cvImage, color, (30,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                x_offset=y_offset=60
                cvImage[y_offset:y_offset+color_bar.shape[0], x_offset:x_offset+color_bar.shape[1]] = color_bar            
            cv2.putText(cvImage, f'{color_name} {self.ml.targetNames[id]}', (150,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            #print(self.ml.targetNames[id]) #debug
        else:
            cv2.putText(cvImage, 'Not detected, please try again!', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow('classified image', cvImage) #showing the card already detected
    
    def executeWindow(self):
        #execute the GUI
        self.configWindow()
        self.window.mainloop()

if __name__ == '__main__':
    window = Gui()
    window.executeWindow()
    