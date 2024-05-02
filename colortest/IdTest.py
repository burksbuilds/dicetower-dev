# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:33:02 2024

@author: aburks
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


templateHueBand = 4 #bilateral
templateSaturationThreshold = 100
templateBrightnessThreshold = 100
templatePolygonFitCriteria = 20
templatePolygonErosionMatrix = np.ones((5,5), np.uint8)
templatePolygonErosionIterations = 10
templateNumeralThreshold = 127
templateNumeralDilationMatrix = np.ones((5,5), np.uint8)
templateNumeralDilationIterations = 10



def getTemplateImage(imageHSV, colorCriteria):
    imageRGB = cv.cvtColor(imageHSV, cv.COLOR_HSV2RGB)
    imageGRAY = cv.cvtColor(imageRGB, cv.COLOR_RGB2GRAY)
    
    masks = [cv.inRange(imageHSV, cc[0], cc[1]) for cc in colorCriteria]
    mask = np.zeros(masks[0].shape, dtype=np.uint8)
    for m in masks:
        mask = cv.add(mask, m)

    
    imageRGB[np.where(mask==0)] = 0
    plt.imshow(imageRGB)
    plt.show()

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contourAreas = [cv.contourArea(c) for c in contours]
    maxContour = np.argmax(contourAreas)
    convexContour = cv.convexHull(contours[maxContour])
    polyContour = cv.approxPolyDP(convexContour, templatePolygonFitCriteria, True)

    mask2 = np.zeros(mask.shape, dtype=np.uint8)
    cv.fillConvexPoly(mask2, polyContour, 255)
    mask2eroded = cv.erode(mask2, templatePolygonErosionMatrix, iterations=templatePolygonErosionIterations)


    imageGRAY[np.where(mask2eroded==0)] = 0
    plt.imshow(imageGRAY, cmap='gray')
    plt.show()


    _,imageNUM = cv.threshold(imageGRAY, templateNumeralThreshold, 255, cv.THRESH_BINARY)

    imageNUMdilated = cv.dilate(imageNUM, templateNumeralDilationMatrix, iterations=templateNumeralDilationIterations)
    rec = cv.boundingRect(cv.findNonZero(imageNUMdilated))
    #imageNUMcrop = np.copy(imageNUM)
    imageNUMcrop = np.copy(imageNUM[rec[1]:rec[1]+rec[3],rec[0]:rec[0]+rec[2]])
    
    cv.rectangle(imageNUM, rec, 255, 3)
    plt.imshow(imageNUM, cmap='gray')
    plt.show()

    
    inverseImage = cv.bitwise_not(imageNUMcrop)

    
    inverseContours, _ = cv.findContours(inverseImage, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    numberZoneArea = rec[2]*rec[3]
    contourAreaThreshold = numberZoneArea * 0.01
    for c in inverseContours:
        contourArea = cv.contourArea(c)
        if contourArea < contourAreaThreshold:
            cv.fillConvexPoly(imageNUMcrop, c, 255)

    plt.imshow(imageNUMcrop, cmap='gray')
    plt.show()
    return imageNUMcrop


def getHueHistogram(imageHSV):
    maskSV = cv.inRange(imageHSV, (0,templateSaturationThreshold,templateBrightnessThreshold), (180,255,255))
    hues = imageHSV[maskSV > 0,0]
    hist,_ = np.histogram(hues,bins=180,range=(0,180),density=True)
    return hist
    
    
    
def getTemplateImagesForDie(rank, fileFormat):
    
    
    dieValues = np.arange(1,rank+1)
    files = [fileFormat.replace("#",str(v)) for v in dieValues]
    imagesHSV = [cv.cvtColor(cv.imread(f),cv.COLOR_BGR2HSV) for f in files]
    
    hist = np.zeros(180, dtype=np.float64)
    for i in imagesHSV:
        hist = np.add(hist,getHueHistogram(i))
    hist = hist / len(imagesHSV)
    hue = int(np.argmax(hist))
    
    hueBand = ((hue-templateHueBand)%180, (hue+templateHueBand)%180)
    colorCriteria = []
    if hueBand[0] > hueBand[1]:
        colorCriteria.append(((0,templateSaturationThreshold,templateBrightnessThreshold),(hueBand[1],255,255)))
        colorCriteria.append(((hueBand[0],templateSaturationThreshold,templateBrightnessThreshold),(180,255,255)))
    else:
        colorCriteria.append(((hueBand[0],templateSaturationThreshold,templateBrightnessThreshold),(hueBand[1],255,255)))
    
    templateImages = [getTemplateImage(i, colorCriteria) for i in imagesHSV]
    

    
fileFormat = "images\\Templates\\template-d6 (#).jpg"
getTemplateImagesForDie(6, fileFormat)
    