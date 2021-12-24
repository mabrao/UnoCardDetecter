# UnoCardDetecter

This is a project by Matheus Abrao.

The main goal of this project is to detect uno cards using computer vision and 
machine learning. This project was written in python using opencv for computer vision
and sklearn for machine learning.

In order to run the project, run the main.py file in terminal using:
'python main.py' or clicking the run code button inside an IDE.

Libraries used were:
- cv2: computer visiom
- os: file management
- numpy: data structures
- pickle: storing data in a file that can be accessed later
- sklearn: classificaition using kmeans clustering and k nearest neighbors
- tkinter: Graphical User Interface (GUI)

if these dependencies are not installed the project cannot be run.


The python files that are used in this project are:

# image_processing.py

The class ImageProcessing performs open cv operations on an image.

When an object of the class is created, it loads the images used for training
and testing. These images are in two separate folders - images and img_descriptor_train.
THe first folder will be used to train the knn classifier for color detection and 
the second folder will be used for getting descriptors of the keypoints found with ORB 
(Oriented Fast and Rotated Brief) for each card.

Inside this class there are methods for applying a threshold, noise removal using morphology,
finding external contours, cropping a card, getting keypoints and descriptors, and finding all
the key descriptors for all diferent cards. There is also a show cards method for showing some of
the operations done on the cards.

# machine_learning.py

The class MachineLearning performs operations using the ORB classifier, Kmeans clustering and K nearest neighbors.

When an object of this class is created, it also creates an object of the ImageProcessing Class and it loads the targets for
the detection of the central logo (list - targetNames).

The method find id returns the index of the list target names in order to determine what is inside the central logo,
this method uses a knn matcher with the ORB algorithm.

The method getVideoStream opens the video stream of the camera of the device running the program and displays a menu 
on how to use the program. If the esc key is pressed, it will detect the central logo using the findID method and detect the color
using getCardColorNew and colorNameDetection.

The method GetCardColorNew returns a bar graph and the BGR of the most dominant color.

The method colorNameDetection uses a Knn classifier to detect the color of the card,
it receives the most dominant color in BGR from the getCardColor new and predicts
which color it is. The dataset that is used to predict the color name is inside the 
pickle file color_classifier.p


# main.py

-RUN THIS FILE!!!

The class Gui inside main.py creates a GUI and executes the code when buttons are pressed.

The GUI has two buttons:
- Detect with camera: Runs the getVideoStream method
- Browse Files: Runs the selectFiles method

The selectFiles method performs central logo detection using ORB (findID method) and then it performs
color detection using getCardColorNew and colorNameDetection.


# reusable_functions.py

These are functions that I wrote but ended up not being used in the project or did not work. 
However, if needed, they can be repurposed.


# Future work that could be added

Performance enhancing:
- Use more and better train images for both the central logo detection and the color detection.
- When the card is not being cropped correctly and it runs through an exception 
(line 158 machine_learning.py), perform kmeans clustering on the entire image instead of
the cropped card, or try and get better results when cropping the card.
- Make a uno game that assumes if the card that was played is correct or not and that 
it tells the players what to do.

# sources

Kmeans clustering
https://www.timpoulsen.com/2018/finding-the-dominant-colors-of-an-image.html

ORB feature detection
https://www.youtube.com/watch?v=nnH55-zD38I



Thank you for taking a look at my project!






