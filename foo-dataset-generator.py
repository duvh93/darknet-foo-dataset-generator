from PIL import Image, ImageDraw
import numpy
import time
import os
import shutil

def fromTopLeftBottomRightToCenterWH(coordinates):
    return [int((coordinates[2]+coordinates[0])/2), \
            int((coordinates[3]+coordinates[1])/2), \
            coordinates[2]-coordinates[0], \
            coordinates[3]-coordinates[1]]

def isIsometric(coordinates):
    return (coordinates[2] - coordinates[0]) == (coordinates[3] - coordinates[1])

def drawRectangle(img, coordinates):
    drawObj = ImageDraw.Draw(img)
    
    randFill = tuple(np.uint8(255*np.random.rand(3)))
    randOutL = tuple(np.uint8(255*np.random.rand(3)))
    drawObj.rectangle(tuple(coordinates), \
                   fill = None if np.random.rand() > 0.5 else randFill, \
                   outline = randOutL)
    
    return "square" if isIsometric(coordinates) else "rectangle"
    
    

def drawEllipse(img, coordinates):
    drawObj = ImageDraw.Draw(img)
    
    randColor = tuple(np.uint8(255*np.random.rand(3)))
    drawObj.ellipse(tuple(coordinates), \
                   fill = randColor, \
                   outline = randColor)    
    
    return "circle" if isIsometric(coordinates) else "ellipse"

def drawTriangle(img, coordinates):
    drawObj = ImageDraw.Draw(img)
    
    topVertexX = int((coordinates[2]+coordinates[0])/2)
    topVertexY = coordinates[1]
    leftVertexX = coordinates[0]
    leftVertexY = coordinates[3]
    rightVertexX = coordinates[2]
    rightVertexY = coordinates[3]
    
    randFill = tuple(np.uint8(255*np.random.rand(3)))
    randOutL = tuple(np.uint8(255*np.random.rand(3)))
    drawObj.polygon(((topVertexX,topVertexY),(leftVertexX,leftVertexY),(rightVertexX,rightVertexY)), \
                   fill = randFill, \
                   outline = randOutL)   
    
    return "triangle"
    
def isOverlapping(proposedBB, listOfBBs):
    overlapped = False
    
    for row in listOfBBs:
         if (row[0]<proposedBB[2]) and \
            (row[2]>proposedBB[0]) and \
            (row[3]>proposedBB[1]) and \
            (row[1]<proposedBB[3]):
            overlapped = True
            break
    
    return overlapped

def proposeBB(width, height, numberOfObjects, isometricProb = 0.1):
    maxWidth = int(width / (numberOfObjects + 1))
    maxHeight = int(height / (numberOfObjects + 1))
    minWidth = int(width / (numberOfObjects + 4))
    minHeight = int(height / (numberOfObjects + 4))

    W = int((maxWidth-minWidth) * np.random.rand() + minWidth)
    H = int((maxHeight-minHeight) * np.random.rand() + minHeight)
    
    if np.random.rand() < isometricProb:
        minDim = min(W,H)
        W, H = minDim, minDim
    
    topLeftCornerX = int((width - W)*np.random.rand())
    topLeftCornerY = int((height - H)*np.random.rand())
    
    return [topLeftCornerX, topLeftCornerY, topLeftCornerX+W, topLeftCornerY+H]

def generateImage(width, height, maxNumberOfObjects, drawingFuncs, classDictionary, emptyImgProb = 0.1, forceMaxNumberOfObjects = False):
    completed = False
    maxAttempts = 100
    listOfBBs = []
    listOfLabels = []
    bgImg = Image.new('RGB', (height, width), (0, 0, 0))
    
    numberOfObjects = maxNumberOfObjects if forceMaxNumberOfObjects else np.random.randint(1,maxNumberOfObjects)
    
    while not completed:
        completed = True
    
        unifNoise = np.uint8(255*np.random.rand(height,width,3))
        bgImg = Image.fromarray( unifNoise )
    
        if np.random.rand() > emptyImgProb:
            for i in range(numberOfObjects):
                iteration = 0
                proposedBB = []
                randomShape = int(np.random.rand()*len(drawingFuncs))
                drwFunc = drawingFuncs[randomShape]
            
                while True:
                    iteration = iteration + 1
                
                    proposedBB = proposeBB(width, height, numberOfObjects)
                
                    if iteration > maxAttempts:
                        #print("MaxNumberOfIterations exceeded!")
                        completed = False
                        break
                
                    if not isOverlapping(proposedBB, listOfBBs):
                        classType = drwFunc(bgImg, [x + y for x, y in zip([+2, +2, -2, -2], proposedBB)])
                        #print("Found a good BoundingBox for " + classType + "!")
                        listOfBBs.append(proposedBB)
                        listOfLabels.append([classDictionary[classType]])
                        break
                    else:
                        pass
                        #print("Trying a new BoundingBox!")
            
                if not completed:
                    listOfBBs = []
                    listOfLabels = []
                    break

    annotation = []
    
    for i in range(0,len(listOfLabels)):
        annotation.append(listOfLabels[i] + listOfBBs[i])
            
    return bgImg, annotation
    
def generateAnnotateAndSaveSyntheticImage(width, height, noo, drawFunctions, classesDict, path):
    img, boundingBoxes = generateImage(width, height, noo, drawFunctions, classesDict)

    annotationMatrixTLBR = (numpy.asarray(boundingBoxes)).astype(float)

    annotationPath = path + ".txt"
    imagePath = path + ".jpg"
    
    #print("Original:")
    #print(annotationMatrixTLBR)

    if annotationMatrixTLBR.size != 0:
        annotationMatrixCtrWH = numpy.column_stack((annotationMatrixTLBR[:,0], numpy.apply_along_axis(fromTopLeftBottomRightToCenterWH, 1, annotationMatrixTLBR[:,1:5])))
        
    #    print("CtrWH:")
    #    print(annotationMatrixCtrWH)

        annotationMatrixCtrWH[:,[1, 3]] = annotationMatrixCtrWH[:,[1, 3]] / width
        annotationMatrixCtrWH[:,[2, 4]] = annotationMatrixCtrWH[:,[2, 4]] / height
    
    #    print("Normalized:")
    #    print(annotationMatrixCtrWH)
        numpy.savetxt(annotationPath, annotationMatrixCtrWH, "%d %.8f %.8f %.8f %.8f")
    else:
        open(annotationPath, 'a').close()
        annotationMatrixCtrWH = annotationMatrixTLBR

    img.save(imagePath, quality = 100)

    return imagePath, annotationPath
    
trainSamples = 10000
validSamples = 2500

newDatasetFolder  = 'dataset'
trainFolder = 'train'
validationFolder = 'validation'
trainsetList = trainFolder + '.txt'
validationsetList = validationFolder + '.txt'

if os.path.isdir(newDatasetFolder):
    shutil.rmtree(newDatasetFolder)

os.mkdir(newDatasetFolder)
os.mkdir(newDatasetFolder + '/' + trainFolder)
os.mkdir(newDatasetFolder + '/' + validationFolder)

drawFunctions = [drawRectangle, drawEllipse, drawTriangle]
noo = 10
width = 736
height = 320

classesDict =	{
  "square": 0,
  "rectangle": 1,
  "ellipse": 2,
  "circle": 3,
  "triangle": 4
}


with open(newDatasetFolder + '/' + trainsetList,'w') as trainListFile:
    for i in range(trainSamples):
        imagePath, annotationPath = generateAnnotateAndSaveSyntheticImage(width, height, noo, drawFunctions, classesDict, newDatasetFolder + '/' + trainFolder + '/trainImg_' + str(i) )
        trainListFile.write(("%s" + ('\n' if i<trainSamples else '') ) % (imagePath))
        
        
with open(newDatasetFolder + '/' + validationsetList,'w') as validListFile:
    for i in range(validSamples):
        imagePath, annotationPath = generateAnnotateAndSaveSyntheticImage(width, height, noo, drawFunctions, classesDict, newDatasetFolder + '/' + validationFolder + '/validImg_' + str(i) )
        validListFile.write(("%s" + ('\n' if i<validSamples else '') ) % (imagePath))
