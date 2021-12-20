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
        filetypes = (
            ('jpg file', '*.jpg'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title= 'Browse files',
            initialdir= './',
            filetypes= filetypes
        )

        print(filename)
        cvImage = cv2.imread(filename)
        cvGrey = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)

        ##Color detection using kmeans clustering##
        #crop card:
        croppedCard = self.ml.ip.cropCards(cvImage)
        #get color
        color = self.ml.getCardColor(croppedCard)

        ## apply ORB on image file
        desList = self.ml.ip.findKeyDes() #get description list of each train image
        id = self.ml.findID(cvGrey, desList) #get id of grey input image

        if id != -1:
            if id not in range(13,17): #if it's not a black special card, put color text on image
                cv2.putText(cvImage, color, (30,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(cvImage, self.ml.targetNames[id], (150,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            #print(self.ml.targetNames[id]) #debug
        else:
            cv2.putText(cvImage, 'Not detected, please try again!', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow('classified image', cvImage)
    
    def executeWindow(self):
        self.configWindow()
        self.window.mainloop()

if __name__ == '__main__':
    window = Gui()
    window.executeWindow()
    
    #ml.videoStream()
    # ml.getVideoStream()
    # ml.ip.findColor(ml.ip.cvImages[35])