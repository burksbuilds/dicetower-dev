import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision.DieFaceTemplate import *
from scipy.spatial.transform import Rotation
import math
import mahotas
from itertools import compress

class DieFaceSample:

    edge_opening_kernel_size = 5
    crop_padding = 25
    MIN_NUMBER_CONTOUR_SIZE = 25 #driving factor for min area is 6vs9 dots on d12
    backround_brightness_threshold = 25
    hue_mask_margin = 20

    def __init__(self, image_cropped_RGBA, geometry, log_level=LOG_LEVEL_FATAL):
        if image_cropped_RGBA.shape[2] != 4:
            raise TypeError("Image must contain 4 channels: RGBA")
        self.image = image_cropped_RGBA
        self.image_hls = cv.cvtColor(self.image[:,:,0:3],cv.COLOR_RGB2HLS)
        self.geometry = geometry
        self.center = (self.image.shape[1]//2, self.image.shape[0]//2)
        self.circumscribed_radius = min(self.center) - DieFaceSample.crop_padding
        self.detector = cv.ORB.create() #cv.SIFT.create()  #cv.AKAZE.create(threshold=0.02)
    
    def get_circular_mask(self, circumscribe_ratio=1.0, log_level=LOG_LEVEL_FATAL):
        mask_radius = int(self.circumscribed_radius * circumscribe_ratio)
        
        mask = np.zeros((self.image.shape[0],self.image.shape[1]), dtype=np.uint8)
        mask = cv.circle(mask,self.center,mask_radius,255,cv.FILLED)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(mask, cmap='gray')
            plt.show()
        return mask
    
    # def view_keypoints(self, keypoints=None, imaging_settings=None):
    #     if imaging_settings is None:
    #         image_debug = cv.cvtColor(self.image,cv.COLOR_RGBA2GRAY)
    #     else:
    #         image_debug = self.__get_keypoint_detection_image(imaging_settings)
    #     if keypoints is None:
    #         keypoints = self.keypoints
    #     image_debug = cv.drawKeypoints(image_debug,keypoints,0,(0,0,255),flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    #     plt.imshow(image_debug,cmap='gray')
    #     plt.title("Keypoints")
    #     plt.show()

    def view_geometry(self):
        image_debug = self.image[:,:,0:3].copy()
        cv.circle(image_debug,self.center,int(self.circumscribed_radius),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["enscribed_perimeter_ratio"]),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["circumscribed_face_ratio"]),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["enscribed_face_ratio"]),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["pixel_comparison_ratio"]),(255,255,255),3)
        plt.imshow(image_debug)
        plt.title("Geometry")
        plt.show()

    # def view_moments(self):
    #     print("Raw Moments:")
    #     for m in self.moments:
    #         print(m[1])
    #     print("Hu Moments:")
    #     for m in self.moments:
    #         print(np.array2string(m[2],max_line_width=32767))
    #     print("Zernike Moments:")
    #     for m in self.moments:
    #         print(np.array2string(m[3], max_line_width=32767))

    def get_sample_mask(self):
        return cv.morphologyEx(self.image[:,:,3],cv.MORPH_ERODE,get_kernel(5),iterations=5)
    

    def get_keypoint_detection_image(self, imaging_settings, log_level=LOG_LEVEL_FATAL):
        #sample_mask = self.get_sample_mask()
        hue_mask = get_hue_mask(self.image_hls[:,:,0],imaging_settings.hue_start,imaging_settings.hue_end, self.hue_mask_margin)

        foreground_mask = self.get_foreground_mask(self.image_hls[:,:,1], hue_mask, log_level=log_level)
        _, sample_mask = self.get_largest_contour_and_mask(foreground_mask, log_level=log_level)
        sample_mask = cv.morphologyEx(sample_mask,cv.MORPH_ERODE,get_kernel(5),iterations=5)
        
        
        _, body_brightness_mask = get_brightness_masks(self.image_hls[:,:,1],imaging_settings.number_brightness, imaging_settings.brightness_threshold)
        numbers_mask = cv.bitwise_not(cv.bitwise_and(hue_mask, body_brightness_mask))    

        image_BW = cv.bitwise_and(numbers_mask,sample_mask) 
        image_BW = cv.morphologyEx(image_BW,cv.MORPH_OPEN,get_kernel(3),iterations=1)
        image_BW = cv.morphologyEx(image_BW,cv.MORPH_CLOSE,get_kernel(3),iterations=1)

        image_contours = draw_all_contours(image_BW,min_area=self.MIN_NUMBER_CONTOUR_SIZE,draw_style=cv.LINE_8)

        if log_level >= LOG_LEVEL_VERBOSE:
            plt.subplot(2,4,1)
            plt.imshow(self.image_hls[:,:,1], cmap='gray')
            plt.title("Brightness")
            plt.subplot(2,4,2)
            plt.imshow(hue_mask, cmap='gray')
            plt.title("Hue Mask")
            plt.subplot(2,4,3)
            plt.imshow(body_brightness_mask, cmap='gray')
            plt.title("Brightness Mask")
            plt.subplot(2,4,4)
            plt.imshow(foreground_mask, cmap='gray')
            plt.title("Foreground Mask")
            
            plt.subplot(2,4,5)
            plt.imshow(sample_mask, cmap='gray')
            plt.title("Sample Mask")
            plt.subplot(2,4,6)
            plt.imshow(numbers_mask, cmap='gray')
            plt.title("Unfiltered")
            plt.subplot(2,4,7)
            plt.imshow(image_BW, cmap='gray')
            plt.title("Filtered")
            plt.subplot(2,4,8)
            plt.imshow(image_contours, cmap='gray')
            plt.title("Contours")
            plt.suptitle("Keypoint Detection Image")
            plt.show()
        return image_contours



    @staticmethod
    def crop_sample_from_image(image_RGB, mask, center, radius, log_level=LOG_LEVEL_FATAL):
        cropX1 = int(max(center[0]-radius-DieFaceSample.crop_padding,0))
        cropX2 = int(min(center[0]+radius+DieFaceSample.crop_padding,image_RGB.shape[1]-1))
        cropY1 = int(max(center[1]-radius-DieFaceSample.crop_padding,0))
        cropY2 = int(min(center[1]+radius+DieFaceSample.crop_padding,image_RGB.shape[0]-1))
        image_Crop_RGBA = cv.cvtColor(image_RGB[cropY1:cropY2,cropX1:cropX2,:],cv.COLOR_RGB2RGBA)
        if mask is not None:
            image_Crop_RGBA[:,:,3] = mask[cropY1:cropY2,cropX1:cropX2]
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.subplot(1,3,1)
            plt.imshow(image_Crop_RGBA[:,:,0:3])
            plt.title("Cropped RGB")
            plt.subplot(1,3,2)
            plt.imshow(image_Crop_RGBA[:,:,3],cmap='gray')
            plt.title("Cropped Alpha Mask")
            plt.subplot(1,3,3)
            plt.gcf().set_size_inches((12,4))
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(image_Crop_RGBA)
            plt.title("Cropped and Masked")
            plt.show()
        return image_Crop_RGBA
    

    @staticmethod
    def get_foreground_mask(brightness, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        _,image_BW = cv.threshold(brightness,DieFaceSample.backround_brightness_threshold,255,cv.THRESH_BINARY)

        if roi_mask is not None:
            image_BW = cv.bitwise_and(image_BW,roi_mask) #image_BW[np.where(roi_mask == 0)] = 0 #only look within the region of interest mask

        # reject noise with an opening morphology (erode then dilate)
        image_Filtered_BW = image_BW.copy()
        image_Filtered_BW = cv.morphologyEx(image_Filtered_BW,cv.MORPH_OPEN,get_kernel(DieFaceSample.edge_opening_kernel_size),iterations=2)
        return image_Filtered_BW
    
    @staticmethod
    def get_largest_contour_and_mask(image_BW, convex_hull=True, aspect_ratio_limit=3, concavity_limit=2, log_level=LOG_LEVEL_FATAL):
        contours, _ = cv.findContours(image_BW,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            if log_level >= LOG_LEVEL_ERROR:
                plt.imshow(image_BW,cmap='gray')
                plt.show()
            raise Exception("Error cropping template image: No contour found")
        contour_properties = [(cv.contourArea(c),cv.minAreaRect(c), cv.convexHull(c)) for c in contours]
        # filte rby aspect artio and convex feature ratio
        contour_Areas = [cp[0] if (cp[1][1][0]/cp[1][1][1] < aspect_ratio_limit and cp[1][1][1]/cp[1][1][0] < aspect_ratio_limit and cv.contourArea(cp[2])/cp[0] < concavity_limit) else 0 for cp in contour_properties]
        largest_contour = contours[contour_Areas.index(max(contour_Areas))]
        if convex_hull:
            largest_contour = cv.convexHull(largest_contour)
        image_Mask_BW = np.zeros(image_BW.shape, dtype=np.uint8)
        cv.drawContours(image_Mask_BW, [largest_contour], 0, 255,thickness=cv.FILLED)
        return largest_contour, image_Mask_BW

    @staticmethod
    def __crop_and_mask_template_image(image_RGB, roi_mask=None, log_level=LOG_LEVEL_FATAL):
    
        image_HLS = cv.cvtColor(image_RGB,cv.COLOR_RGB2HLS)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.subplot(1,4,1)
            plt.imshow(image_RGB)
            plt.title("Original")
            plt.subplot(1,4,2)
            plt.imshow(image_HLS[:,:,0],cmap='gray')
            plt.title("Hue")
            plt.subplot(1,4,3)
            plt.imshow(image_HLS[:,:,2],cmap='gray')
            plt.title("Saturation")
            plt.subplot(1,4,4)
            plt.imshow(image_HLS[:,:,1],cmap='gray')
            plt.title("Brightness")
            plt.gcf().set_size_inches((12,4))
            plt.suptitle("Input Image")
            plt.show()
        #1) Threshold saturation to get a grayscale image that rejects shadows and highlight the one die we expect near the middle
        foreground_mask = DieFaceSample.get_foreground_mask(image_HLS[:,:,1], roi_mask)

        #3) find the biggest contour and create a convex hull around it. This is the die.
        largest_contour, image_Mask_BW = DieFaceSample.get_largest_contour_and_mask(foreground_mask, log_level=log_level)

        #4) circumscribe the convext hull to get the center and rotation-invariant "extents" of the die
        (x,y), r = cv.minEnclosingCircle(largest_contour)
        center = (int(x),int(y))
        radius = int(r)
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.subplot(1,3,1)
            plt.imshow(foreground_mask,cmap='gray')
            plt.title("Filtered")
            plt.subplot(1,3,2)
            plt.imshow(image_Mask_BW,cmap='gray')
            plt.title("Die Contour")
            image_ID = image_RGB.copy()
            cv.drawContours(image_ID,[largest_contour],0,(255,255,255),3)
            cv.circle(image_ID,center,radius,(255,0,0),3)
            plt.subplot(1,3,2)
            plt.imshow(image_ID)
            plt.title("Die Highlight")
            plt.suptitle("Die Extraction")
            plt.gcf().set_size_inches((8,4))
            plt.show()

        #5) crop around the circle (with padding) to get a scale-invariant image of the die. move the mask into the alpha channel
        
        return DieFaceSample.crop_sample_from_image(image_RGB, image_Mask_BW,center,radius,log_level)

    @staticmethod
    def __create_from_uncropped_image(image_or_file, geometry, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        if not isinstance(image_or_file, np.ndarray):
            image_BGR = cv.imread(image_or_file)
            if image_BGR is None:
                if log_level>=LOG_LEVEL_ERROR:
                    print("ERROR: FILE NOT FOUND! Verify that relative paths join properly")
                    print(image_or_file)
                    print(os.getcwd())
                raise FileNotFoundError()
            image_RGB = cv.cvtColor(image_BGR,cv.COLOR_BGR2RGB)
        else:
            image_RGB = image_or_file
            if image_RGB.shape[2] != 3:
                raise Exception("Image must contain at least three channels")
        return DieFaceSample(DieFaceSample.__crop_and_mask_template_image(image_RGB, roi_mask, log_level), geometry, log_level=log_level)
    
    @staticmethod
    def __create_from_cropped_image(image_or_file, geometry, log_level=LOG_LEVEL_FATAL):
        if not isinstance(image_or_file, np.ndarray):
            image_raw = cv.imread(image_or_file,cv.IMREAD_UNCHANGED)
            if image_raw is None:
                if log_level>=LOG_LEVEL_ERROR:
                    print("ERROR: FILE NOT FOUND! Verify that relative paths join properly")
                    print(image_or_file)
                    print(os.getcwd())
                raise FileNotFoundError()
        else:
            image_raw = image_or_file
        if image_raw.shape[2] < 3:
            raise Exception("Image must contain RGB channels, optionally Alpha")
        if image_raw.shape[2] == 3:
            return DieFaceSample(cv.cvtColor(image_raw,cv.COLOR_BGR2RGBA), geometry, log_level)
        if image_raw.shape[2] == 4:
            return DieFaceSample(cv.cvtColor(image_raw,cv.COLOR_BGRA2RGBA), geometry, log_level)
        else:
            raise Exception("Image must contain RGB channels, optionally Alpha")
        
    @staticmethod
    def create_from_image(image_or_file, geometry, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        if autocrop:
            return DieFaceSample.__create_from_uncropped_image(image_or_file, geometry, roi_mask, log_level)
        else:
            return DieFaceSample.__create_from_cropped_image(image_or_file, geometry, log_level)
        
    @staticmethod
    def create_from_image_files(image_files, geometry, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        if os.path.isdir(image_files):
            dir_files = [f for f in os.listdir(image_files) if os.path.isfile(os.path.join(image_files,f))]
            return [DieFaceSample.create_from_image(f, geometry,  autocrop, roi_mask, log_level) for f in dir_files]
        else:
            return [DieFaceSample.create_from_image(f, geometry,  autocrop, roi_mask, log_level) for f in image_files]
        

    def create_from_image_file_template(image_file_template, geometry,autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        images_RGB = get_mactching_images_RGB(image_file_template, log_level)
        samples = [DieFaceSample.create_from_image(i, geometry, autocrop, roi_mask, log_level) for i in images_RGB]
        return samples

