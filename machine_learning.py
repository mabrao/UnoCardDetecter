import image_processing
import numpy as np
import cv2
import pickle
from reusable_functions import getCardColor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
#import matplotlib.pyplot as plt


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

    def getVideoStream(self):
        vc = cv2.VideoCapture(0)
        desList = self.ip.findKeyDes()

        while True:
            rval, frame = vc.read()
            imgOriginal = frame.copy()
            imgDetected = frame.copy()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            
            cv2.putText(imgOriginal, '-Press esc to detect card', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(imgOriginal, '-Press space to end program', (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('frame', imgOriginal)
            
            key = cv2.waitKey(1)
            
            if (key == 27):
                cv2.imwrite('./test_images/saved.jpg', imgDetected)
                saved = cv2.imread('./test_images/saved.jpg')

                id = self.findID(saved, desList)

                croppedCard = self.ip.cropCards(saved)

                #color = self.ip.findColor(imgOriginal)

                #get color
                #print(croppedCard.shape)
                color_bar, _ = self.getCardColorNew(croppedCard)
                #find color name of cropped card:
                color_name = self.colorNameDetection(croppedCard)


                if id != -1:
                    if id not in range(13,17): #if it's not a black special card, put color text on image
                        #cv2.putText(saved, color, (30,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                        x_offset=y_offset=60
                        saved[y_offset:y_offset+color_bar.shape[0], x_offset:x_offset+color_bar.shape[1]] = color_bar  
                        cv2.putText(saved, f'{color_name} {self.targetNames[id]}', (150,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(saved, f'{self.targetNames[id]}', (150,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    #print(self.targetNames[id])
                else:
                    cv2.putText(saved, 'Not detected, please try again!', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

                
                cv2.imshow('classified image', saved)
                
                
            if (key == 32): #press space to end program
                cv2.destroyAllWindows()
                vc.release()
                break
    def getCardColorNew(self,croppedCard):
        
        def find_histogram(clt):
            """
            create a histogram with k clusters
            :param: clt
            :return:hist
            """
            numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
            (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        
            hist = hist.astype("float")
            hist /= hist.sum()
        
            return hist
        def plot_colors2(hist, centroids):
            bar = np.zeros((50, 300, 3), dtype="uint8")
            startX = 0
        
            for (percent, color) in zip(hist, centroids):
                # plot the relative percentage of each cluster
                endX = startX + (percent * 300)
                cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                              color.astype("uint8").tolist(), -1)
                startX = endX

            #find index of most dominant color:
            most_dominant_index = np.where(hist==max(hist))
            
            #find most dominant color in BGR:
            #print(centroids[most_dominant_index][0]) #debug
            b,g,r = int(centroids[most_dominant_index][0][0]), int(centroids[most_dominant_index][0][1]), int(centroids[most_dominant_index][0][2])
            
            most_dominant_color = (b,g,r)

            # return the bar chart
            return bar, most_dominant_color
        
        #img = croppedCard#cv2.imread("pic/img7.jpeg")
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = croppedCard.reshape((croppedCard.shape[0] * croppedCard.shape[1],3)) #represent as row*column,channel number
        #print(img.shape)
        clt = KMeans(n_clusters=3) #cluster number

        try:
            clt.fit(img)
            hist = find_histogram(clt)
            bar, color_bgr = plot_colors2(hist, clt.cluster_centers_)
            return bar, color_bgr

        except Exception as e:
            print(f'Please close the windows and try again, card is not being cropped correctly - {e}')
        
        
    
    ##K nearest neighbours for name of the color detection
    def colorNameDetection(self, croppedCard):
        '''
        Color name detection with Knn classifier.
        '''
        def writeClassifier():
            #load x train data:
            x_train = []
            #load y train data, blue = 0, green = 1, black = 2, red = 3, yellow = 4
            y_train = [
                0,0,0,0,0,0,0,0,0,0,0,0,0,
                1,1,1,1,1,1,1,1,1,1,1,1,1,
                2,2,2,2,
                3,3,3,3,3,3,3,3,3,3,3,3,3,
                4,4,4,4,4,4,4,4,4,4,4,4,4,
            ]
            for image in self.ip.cvImages:
                cropped = self.ip.cropCards(image)
                _, (b,g,r) = self.getCardColorNew(cropped)
                x_train.append((b,g,r))


            #create classifier:
            clf = KNeighborsClassifier(2)
            clf.fit(x_train, y_train)

            #save the model in a pcikle file after training
            pickle.dump(clf, open('color_classifier.p', 'wb'))
        
        #writeClassifier() #write the classifier if more training data is acquired

        clf = pickle.load(open('color_classifier.p', 'rb')) #load the model to test it

        _, (b,g,r) = self.getCardColorNew(croppedCard) 
        #print((b,g,r))
        y_predict = clf.predict([(b,g,r)])
        
        if y_predict[0] == 0:
            return 'blue'
        elif y_predict[0] == 1:
            return 'green'
        elif y_predict[0] == 2:
            return 'black'
        elif y_predict[0] == 3:
            return 'red'
        elif y_predict[0] == 4:
            return 'yellow'
        else:
            return 'name of color not detected'

        
    

    


    



