# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:33:02 2024

@author: aburks
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

showDebugPlots = True

minimumDieContourArea = 1000 # pixels?

templateHueBandThreshold = 0.99
templateSaturationThreshold = 50
templateBrightnessThreshold = 50
templatePolygonFitCriteria = 20
templatePolygonErosionMatrix = np.ones((3,3), np.uint8)
templatePolygonErosionIterations = 10
templateNumeralThreshold = 127
templateNumeralDilationMatrix = np.ones((3,3), np.uint8)
templateNumeralDilationIterations = 10
externalFeatureAreaThreshold = 0.005
externalFeatureDistanceThreshold = 0.5
internalFeatureAreaThreshold = 0.01


def extractMatchingDiceRGB(imageHSV, colorCriteria, focusScale):
    if showDebugPlots:
        plt.imshow(imageHSV, cmap='hsv')
        plt.show() 
    masks = [cv.inRange(imageHSV, cc[0], cc[1]) for cc in colorCriteria]
    mask = np.zeros(masks[0].shape, dtype=np.uint8)
    for m in masks:
        mask = cv.add(mask, m)
    if showDebugPlots:
        plt.imshow(mask, cmap='gray')
        plt.show() 
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contourAreas = [cv.contourArea(c) for c in contours]
    
    orderedContours = np.flip(np.argsort(contourAreas))
    
    #maxContourArea = max(contourAreas)
    #contourThreshold = maxContourArea * 0.5
    
    for c in orderedContours:
        contour = contours[c]
        if cv.contourArea(contour) < minimumDieContourArea:
            continue
        
        #convexContour = cv.convexHull(contour)
        #polyContour = cv.approxPolyDP(convexContour, templatePolygonFitCriteria, True)
        #polyMask = np.zeros(mask.shape,dtype=np.uint8)
        #cv.drawContours(polyMask, [polyContour], -1, 255,thickness=cv.FILLED)
        
        (centerX,centerY),radius = cv.minEnclosingCircle(contour)
        focusRadius = int(radius*focusScale)
        circMask = np.zeros(mask.shape,dtype=np.uint8)
        cv.circle(circMask, (int(centerX),int(centerY)), int(focusRadius), 255,thickness=cv.FILLED)
        
        cropT = int(centerY-focusRadius)
        cropB = int(centerY+focusRadius)
        cropL = int(centerX-focusRadius)
        cropR = int(centerX+focusRadius)
        imageCrop = np.copy(imageHSV[cropT:cropB,cropL:cropR,:])
        maskCrop = np.copy(circMask[cropT:cropB,cropL:cropR])
        imageCrop[np.where(maskCrop==0)] = 0
        imageCropRGB = cv.cvtColor(imageCrop, cv.COLOR_HSV2RGB)
        if showDebugPlots:
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
        print(cv.contourArea(c))
        if cv.contourArea(c) < areaThreshold:
            print("too small")
            continue
        extendsPastThreshold = [abs(p[:,0] - centerX) > distanceThreshold or abs(p[:,1] - centerY) > distanceThreshold for p in c]
        if np.any(extendsPastThreshold) :
            print("too far away")
            continue
        print("contour passed!")
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
    return imageBW
    


def getTemplateImage(imageHSV, colorCriteria, erosion, show):
    print("getting template image")
    matchingDiceRGB = extractMatchingDiceRGB(imageHSV, colorCriteria, 0.5)
    return getNumberFeatureBW(next(matchingDiceRGB),0.9)


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
    if hueBand[0] > hueBand[1]:
        colorCriteria.append(((0,templateSaturationThreshold,templateBrightnessThreshold),(hueBand[1],255,255)))
        colorCriteria.append(((hueBand[0],templateSaturationThreshold,templateBrightnessThreshold),(180,255,255)))
    else:
        colorCriteria.append(((hueBand[0],templateSaturationThreshold,templateBrightnessThreshold),(hueBand[1],255,255)))
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
    colorCriteria = [((0,0,0),(180,255,threshold))]
    return colorCriteria

def getColorCriteria(imagesHSV):
    hueHistogram = getHueHistogram(imagesHSV)
    hueBand = getHueBand(hueHistogram,0.99)
    
    brightnessHistogram = getBrightnessHistogram(imagesHSV)
    brightnessThreshold = getBrightnessThreshold(brightnessHistogram, 0.01)
    
    if hueBand[1] < 15:
        print("using hue band for colored die")
        return getColorCriteriaHue(hueBand)
    else:
        print("using brightness band for greyscale die")
        return getColorCriteriaBrightness(brightnessThreshold)
    
def getTemplateInfoForDie(rank, fileFormat, erosion):
    
    
    dieValues = np.arange(1,rank+1)
    dieFileFormat = fileFormat.replace("D#","d"+str(rank))
    files = [dieFileFormat.replace("V#",str(v)) for v in dieValues]
    imagesHSV = [cv.cvtColor(cv.imread(f),cv.COLOR_BGR2HSV) for f in files]
    
    colorCriteria = getColorCriteria(imagesHSV)
    
    templateImages = [getTemplateImage(i, colorCriteria, erosion, True) for i in imagesHSV]
    return (colorCriteria, templateImages)

    
fileFormat = "images\\Templates\\template-D# (V#).jpg"
getTemplateInfoForDie(8, fileFormat, 25)
    