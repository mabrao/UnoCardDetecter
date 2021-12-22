#code used for computing all bounds (not nescessary to run every time)
#easier to just hard code the bounds after finding them
def computeColorBounds(self):
    rL, gL, bL = [], [], []
    for i, image in enumerate(self.ip.cvImages):
        try:
            r,g,b = self.colorSeparation(image)
        except:
            print(self.colorSeparation(image))
        rL.append(r)
        gL.append(g)
        bL.append(b)
        if (i == 12): #computing blue bound
            #print(rL, gL, bL)
            blueLowerBound = [min(rL), min(gL), min(bL)]
            blueUpperBound = [max(rL), max(gL), max(bL)]
            print(f'{blueLowerBound = } {blueUpperBound = }')
            rL, gL, bL = [], [], []
        elif (i == 25): #computing green bound
            greenLowerBound = [min(rL), min(gL), min(bL)]
            greenUpperBound = [max(rL), max(gL), max(bL)]
            print(f'{greenLowerBound = } {greenUpperBound = }')
            rL, gL, bL = [], [], []
        elif (i == 29): #ignore black (special) cards
            rL, gL, bL = [], [], []
        elif (i == 42): #compunting red bound
            redLowerBound = [min(rL), min(gL), min(bL)]
            redUpperBound = [max(rL), max(gL), max(bL)]
            print(f'{redLowerBound = } {redUpperBound = }')
            rL, gL, bL = [], [], []
        elif (i == 55): #computing yellow bound
            yellowLowerBound = [min(rL), min(gL), min(bL)]
            yellowUpperBound = [max(rL), max(gL), max(bL)]
            print(f'{yellowLowerBound = } {yellowUpperBound = }')
            rL, gL, bL = [], [], []