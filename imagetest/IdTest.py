# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:33:02 2024

@author: aburks
"""

import machinevisiontoolbox as mvt
import os

if 'distortionParameters' not in globals():
    print("Calculating New Distortion Parameters")
    calibrationImageFiles = os.path.join('images','calibration','*.jpg')
    calibrationImages = mvt.ImageCollection(calibrationImageFiles)
    
    camera = mvt.CentralCamera(f=0.008)
    (cameraCalibrationMatrix, distortionParameters, imageFrames) = camera.images2C(calibrationImages, gridshape=(7,5), squaresize=0.029)
    
else:
    print("Re-Using Old Distortion Parameters")


templateImages_d6 = mvt.ImageCollection(os.path.join('images','templates','d6*.jpg'))

templateImages_d6_corrected = [i.undistort(cameraCalibrationMatrix, distortionParameters) for i in templateImages_d6]

templateImages_d6[0].disp()
templateImages_d6_corrected[0].disp()

testImages_d6d8 = mvt.ImageCollection(os.path.join('images','tests','d6d8*.jpg'))
testImages_d6d8_corrected = [i.undistort(cameraCalibrationMatrix, distortionParameters) for i in testImages_d6d8]