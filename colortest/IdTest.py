 #%%
 # # -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:33:02 2024

@author: aburks
"""

import cv2 as cv
import numpy as np
#import matplotlib
#matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from collections import namedtuple

showDebugPlots = False

minimumDieContourArea = 1000 # pixels?

templateHueBandThreshold = 0.95
templateSaturationThreshold = 50
templateBrightnessThreshold = 25
# templatePolygonFitCriteria = 20
# templatePolygonErosionMatrix = np.ones((3,3), np.uint8)
# templatePolygonErosionIterations = 10
# templateNumeralThreshold = 127
# templateNumeralDilationMatrix = np.ones((3,3), np.uint8)
# templateNumeralDilationIterations = 10
externalFeatureAreaThreshold = 0.001
externalFeatureDistanceThreshold = 0.5
internalFeatureAreaThreshold = 0.01
contourMaskErosion = 25

ValueTemplate = namedtuple('ValueTemplate',['value','imageTemplateBW'])
DieSettings = namedtuple('DieSettings',['erosion', 'focusScale', 'colorCriteria'])
DieTemplate = namedtuple('DieTemplate',['rank','settings','valueTemplates'])


def extractMatchingDiceRGB(imageHSV, dieSettings):
    if showDebugPlots:
        plt.imshow(imageHSV, cmap='hsv')
        plt.show() 
    masks = [cv.inRange(imageHSV, cc[0], cc[1]) for cc in dieSettings.colorCriteria]
    mask = np.zeros(masks[0].shape, dtype=np.uint8)
    for m in masks:
        mask = cv.add(mask, m)
    if showDebugPlots:
        plt.imshow(mask, cmap='gray')
        plt.show() 
    blurredMask = cv.GaussianBlur(mask, (51,51), 0)
    blurredMask = cv.inRange(blurredMask, 50, 255)
    if showDebugPlots:
        plt.imshow(blurredMask, cmap='gray')
        plt.show() 
    contours, _ = cv.findContours(blurredMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contourAreas = [cv.contourArea(c) for c in contours]
    
    orderedContours = np.flip(np.argsort(contourAreas))
    
    #maxContourArea = max(contourAreas)
    #contourThreshold = maxContourArea * 0.5
    
    for c in orderedContours:
        contour = contours[c]
        if cv.contourArea(contour) < minimumDieContourArea:
            return
        
        #convexContour = cv.convexHull(contour)
        #polyContour = cv.approxPolyDP(convexContour, templatePolygonFitCriteria, True)
        contourMask = np.zeros(mask.shape,dtype=np.uint8)
        cv.drawContours(contourMask, [contour], -1, 255,thickness=cv.FILLED)
        for i in np.arange(0, dieSettings.erosion):
            contourMask = cv.erode(contourMask, np.ones((3, 3), np.uint8))  
        if showDebugPlots:
            plt.imshow(contourMask, cmap='gray')
            plt.show()
        

        (centerX,centerY),radius = cv.minEnclosingCircle(contour)
        focusRadius = int(radius-contourMaskErosion)
        circMask = np.zeros(mask.shape,dtype=np.uint8)
        cv.circle(circMask, (int(centerX),int(centerY)), int(focusRadius), 255,thickness=cv.FILLED)
        
        cropT = int(centerY-focusRadius)
        cropB = int(centerY+focusRadius)
        cropL = int(centerX-focusRadius)
        cropR = int(centerX+focusRadius)
        imageRGB = cv.cvtColor(imageHSV,cv.COLOR_HSV2RGB)
        imageRGB[np.where(circMask==0)] = 0
        imageRGB[np.where(contourMask==0)] = 0
        imageCropRGB = np.copy(imageRGB[cropT:cropB,cropL:cropR,:])
        
        plt.imshow(imageCropRGB)
        plt.show()
        yield imageCropRGB
        
        
def getNumberFeatureBW(imageRGB, focusScale):
    imageHSV = cv.cvtColor(imageRGB,cv.COLOR_RGB2HSV)
    imageGRAY = cv.cvtColor(imageRGB, cv.COLOR_RGB2GRAY)
    histGRAY,_ = np.histogram(imageGRAY[imageGRAY > 0],bins=256,range=(0,256),density=True) # exclude outside of circle
    thresholdBW = np.min([255,np.max(np.nonzero(histGRAY > 0.02))+1]) # TODO magic number, consider kmeans
    #_,imageBW = cv.threshold(imageGRAY, thresholdBW, 255, cv.THRESH_BINARY)
    imageBW = cv.inRange(imageHSV, (0,0,50),(180,100,255))
    numberMask = np.zeros(imageBW.shape,dtype=np.uint8) # mask of all contours with numeral features
    if showDebugPlots:
        plt.imshow(imageHSV, cmap='hsv')
        plt.show() 
        plt.imshow(imageGRAY, cmap='gray')
        plt.show() 
        plt.imshow(imageBW, cmap='gray')
        plt.show()
        #plt.imshow(imageBW2, cmap='gray')
        #plt.show()
    
    # REMOVE ALL WHITE FEATURES THAT ARE TOO SMALL OR TOO FAR FROM CENTER TO BE A NUMERAL
    contours, _ = cv.findContours(imageBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    distanceThreshold = imageBW.shape[0]/2 * focusScale
    areaThreshold = imageBW.shape[0]*imageBW.shape[1]*externalFeatureAreaThreshold #0.01 is too big and filters out the dot next to '6'
    centerX = int(imageBW.shape[0]/2)
    centerY = int(imageBW.shape[1]/2)
    
    for c in contours:
        if cv.contourArea(c) < areaThreshold:
            #print("ftoo small: {cv.contourArea(c)} < {areaThreshold}")
            continue
        extendsPastThreshold = [abs(p[:,0] - centerX) > distanceThreshold or abs(p[:,1] - centerY) > distanceThreshold for p in c]
        if np.any(extendsPastThreshold) :
            #print("too far away")
            continue
        #print("contour passed!")
        cv.drawContours(numberMask, [c], -1, 255,thickness=cv.FILLED)
    imageBW[numberMask == 0] = 0 #delete all white features outside identified contours (retains black features isnide white features)
    if showDebugPlots:
        plt.imshow(numberMask, cmap='gray')
        plt.show() 
        plt.imshow(imageBW, cmap='gray')
        plt.show()
    
    # REMOVE BLACK FEATURES CONTAINED IN WHITE FEATURE
    
    rec = cv.boundingRect(cv.findNonZero(imageBW))
    numberZoneArea = rec[2]*rec[3]
    contourAreaThreshold = numberZoneArea * internalFeatureAreaThreshold
    inverseImage = cv.bitwise_not(imageBW)
    inverseContours, _ = cv.findContours(inverseImage, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for c in inverseContours:
        contourArea = cv.contourArea(c)
        if contourArea < contourAreaThreshold:
            cv.drawContours(imageBW, [c], -1, 0, thickness=cv.FILLED)
    
    if showDebugPlots:
        plt.imshow(imageBW, cmap='gray')
        plt.show()
    
    imageBWcrop = imageBW[rec[1]-1:rec[1]+rec[3]+2,rec[0]-1:rec[0]+rec[2]+2]
    plt.imshow(imageBWcrop, cmap='gray')
    plt.show()
    
    return imageBWcrop
    


def getTemplateImage(imageHSV, dieSettings):
    print("getting template image...")
    matchingDiceRGB = extractMatchingDiceRGB(imageHSV, dieSettings)
    return getNumberFeatureBW(next(matchingDiceRGB),dieSettings.focusScale)


def getHueHistogram(imageHSV):
    if type(imageHSV) is list: #case where list of images
        hist = np.zeros(180, dtype=np.float64)
        for i in imageHSV:
            hist = np.add(hist,getHueHistogram(i))
        hist = hist / len(imageHSV)
        return hist
    else:
        maskSV = cv.inRange(imageHSV, (0,templateSaturationThreshold,templateBrightnessThreshold), (180,255,255))
        hues = imageHSV[maskSV > 0,0]
        if len(hues) == 0:
            return np.ones(180, dtype=np.float64)/180
        hist,_ = np.histogram(hues,bins=180,range=(0,180),density=True)
        return hist
    
def getHueBand(hueHistogram, coverage):
    # TODO consider replacing this with k-means clustering
    hueTotals = np.copy(hueHistogram)
    bandWidth = 1
    while np.max(hueTotals) < coverage :
        for i in np.arange(0,len(hueHistogram)):
            hueTotals[i] = hueTotals[i] + hueHistogram[(i+bandWidth)%len(hueHistogram)]
        bandWidth = bandWidth + 1
    hueStart = int(np.argmax(hueTotals))
    #return (hueStart,(hueStart+bandWidth)%len(hueHistogram))
    return (hueStart, bandWidth)

def getColorCriteriaKmeans(imagesHSV):
    imagesRGB = [cv.cvtColor(i,cv.COLOR_HSV2RGB) for i in imagesHSV]
    pixelsRGB = np.float32(np.vstack( [i.reshape((-1,3)) for i in imagesRGB]))
    ret,label,center = cv.kmeans(pixelsRGB,2,None,(),10,cv.KMEANS_RANDOM_CENTERS)
    if np.sum(center[0]) > np.sum(center[1]) :
        bg = center[0]
        fg = center[1]
    else:
        bg = center[1]
        fg = center[0]
    lower = (0,0,0)
    upper = (255,255,255)
    for i in range(3):
        if fg[i] > bg[i]:
            upper[i] = int((bg[i]+fg[i])/2)
        else:
            lower[i] = int((bg[i]+fg[i])/2)
    return [(lower,upper)]
    
    
    
def getColorCriteriaHue(hueBand):
    colorCriteria = []
    hueBandStart = hueBand[0]
    hueBandEnd = (hueBand[0]+hueBand[1]-1)%180
    if hueBandStart > hueBandEnd:
        colorCriteria.append(((0,templateSaturationThreshold,templateBrightnessThreshold),(hueBandEnd,255,255)))
        colorCriteria.append(((hueBandStart,templateSaturationThreshold,templateBrightnessThreshold),(180,255,255)))
    else:
        colorCriteria.append(((hueBandStart,templateSaturationThreshold,templateBrightnessThreshold),(hueBandEnd,255,255)))
    return colorCriteria

def getBrightnessHistogram(imageHSV):
    if type(imageHSV) is list: #case where list of images
        hist = np.zeros(256, dtype=np.float64)
        for i in imageHSV:
            hist = np.add(hist,getBrightnessHistogram(i))
        hist = hist / len(imageHSV)
        return hist
    hist,_ = np.histogram(imageHSV[:,:,2],bins=256,range=(0,256),density=True)
    return hist
    
def getBrightnessThreshold(brightnessHistogram, coverage):
    total = 0
    i = 0
    while total < coverage and i < len(brightnessHistogram):
        total = total + brightnessHistogram[i]
        i = i + 1
    return i
    

def getColorCriteriaBrightness(threshold):
    colorCriteria = [((0,templateSaturationThreshold,0),(180,255,threshold))]
    return colorCriteria

def getColorCriteria(imagesHSV):
    hueHistogram = getHueHistogram(imagesHSV)
    hueBand = getHueBand(hueHistogram,templateHueBandThreshold)
    
    brightnessHistogram = getBrightnessHistogram(imagesHSV)
    brightnessThreshold = getBrightnessThreshold(brightnessHistogram, 0.01)
    
    if hueBand[1] < 15:
        print(f"using hue band [{hueBand[0]},{(hueBand[0]+hueBand[1]-1)%180}] for colored die")
        plt.figure()
        plt.bar(np.arange(0, 180),hueHistogram)
        plt.show()
        return getColorCriteriaHue(hueBand)
    else:
        print(f"using brightness band [0,{brightnessThreshold}] for greyscale die")
        return getColorCriteriaBrightness(brightnessThreshold)
    
def getTemplateInfoForDie(rank, fileFormat, erosion, focusScale, outputFileFormat):
    

    
    dieValues = np.arange(1,rank+1)
    dieFileFormat = fileFormat.replace("D#","d"+str(rank))
    files = [dieFileFormat.replace("V#",str(v)) for v in dieValues]
    imagesHSV = [cv.cvtColor(cv.imread(f),cv.COLOR_BGR2HSV) for f in files]
    
    colorCriteria = getColorCriteria(imagesHSV)
    dieSettings = DieSettings(erosion,focusScale,colorCriteria)
    
    templateImages = [ValueTemplate(value,exportTemplateImage(getTemplateImage(image, dieSettings),rank,value,outputFileFormat)) for image,value in zip(imagesHSV,dieValues)]
    
    return DieTemplate(rank,dieSettings,templateImages)

def exportTemplateImage(image, rank, value, imageFileFormat):
    imageFileName = imageFileFormat.replace("D#",str(rank)).replace("V#",str(value))
    cv.imwrite(imageFileName, image)
    return imageFileName


    

    
inputImageFileFormat = "Images\\Templates\\template-D# (V#).jpg"
outputImageFileFormat = "templates\\dD#-V#.bmp"

d6 = getTemplateInfoForDie(6, inputImageFileFormat, 25, 0.75, outputImageFileFormat)
d8 = getTemplateInfoForDie(8, inputImageFileFormat, 25, 0.5, outputImageFileFormat)
d10 = getTemplateInfoForDie(10, inputImageFileFormat, 25, 0.6, outputImageFileFormat)
d12 = getTemplateInfoForDie(12, inputImageFileFormat, 25, 0.5, outputImageFileFormat)
d20 = getTemplateInfoForDie(20, inputImageFileFormat, 25, 0.4, outputImageFileFormat)

    
# %%
