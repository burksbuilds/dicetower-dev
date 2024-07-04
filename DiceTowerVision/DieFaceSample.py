import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *

class DieFaceSample:

    edge_opening_kernel_size = 3
    crop_padding = 25


    
    def get_circular_mask(self, circumscribe_ratio, log_level=LOG_LEVEL_FATAL):
        mask_radius = int(self.circumscribed_radius * circumscribe_ratio)
        
        mask = np.zeros((self.image.shape[0],self.image.shape[1]), dtype=np.uint8)
        mask = cv.circle(mask,self.center,mask_radius,255,cv.FILLED)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(mask, cmap='gray')
            plt.show()
        return mask
    
    def view_keypoints(self):
        image_debug = cv.drawKeypoints(self.image,self.keypoints,0,(0,0,255),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_debug)
        plt.title("Keypoints")
        plt.show()

    def view_geometry(self):
        image_debug = self.image[:,:,0:3].copy()
        cv.circle(image_debug,self.center,int(self.circumscribed_radius),(0,0,0),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["enscribed_perimeter_ratio"]),(0,0,0),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["circumscribed_face_ratio"]),(0,0,0),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["enscribed_face_ratio"]),(0,0,0),3)
        plt.imshow(image_debug)
        plt.title("Geometry")
        plt.show()

    def __update_keypoints(self, log_level=LOG_LEVEL_FATAL):
        mask = self.get_circular_mask(self.geometry["enscribed_perimeter_ratio"])
        self.keypoints, self.descriptors = self.detector.detectAndCompute(cv.cvtColor(self.image,cv.COLOR_RGBA2GRAY),mask)
        if log_level>=LOG_LEVEL_DEBUG:
            self.view_keypoints()

    def __init__(self, image_cropped_RGBA, geometry, log_level=LOG_LEVEL_FATAL):
        if image_cropped_RGBA.shape[2] != 4:
            raise TypeError("Image must contain 4 channels: RGBA")
        self.image = image_cropped_RGBA
        self.geometry = geometry
        self.center = (self.image.shape[1]//2, self.image.shape[0]//2)
        self.circumscribed_radius = min(self.center) - DieFaceSample.crop_padding
        self.detector = cv.AKAZE.create()
        self.__update_keypoints(log_level)

    def compare_to(self, other, log_level=LOG_LEVEL_FATAL):
        results = dict()

        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(other.descriptors, self.descriptors, k=2)
        good_matches = [m for (m,n) in matches if m.distance < 0.75*n.distance]

        results["template_sample"] = self
        results["template_keypoints"] = np.count_nonzero(self.keypoints)
        results["sample_keypoints"] = np.count_nonzero(other.keypoints)
        results["total_matches"] = np.count_nonzero(matches)
        results["good_matches"] = np.count_nonzero(good_matches)

        

        if log_level>=LOG_LEVEL_DEBUG:
            comparison = cv.drawMatches(other.image,other.keypoints,self.image,self.keypoints,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(comparison)
            plt.show()

        if results["good_matches"] <= 4:
            results["corrected_image"] = None
            results["homography"] = None
            results["used_matches"] = 0
            return results

        other_points = np.float32([ other.keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        self_points = np.float32([ self.keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

        homography, mask = cv.findHomography(other_points, self_points, cv.RANSAC, 3.0)
        if homography is None or len(mask) == 0:
            results["corrected_image"] = None
            results["homography"] = None
            results["used_matches"] = 0
            return results


        matched_image = cv.warpPerspective(other.image,homography,(other.image.shape[1],other.image.shape[0]))
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(matched_image)
            plt.show()
        
        
        results["corrected_image"] = matched_image
        results["homography"] = homography
        results["used_matches"] = np.count_nonzero(mask)

        
        _, rotation, translation, normals = cv.decomposeHomographyMat(homography,np.eye(3))
        #results["translation"] = np.average([np.sqrt(np.sum(np.power(t,2))) for t in translation])

        return results

    
    

    @staticmethod
    def crop_sample_from_image(image_RGB, mask, center, radius, log_level=LOG_LEVEL_DEBUG):
        cropX1 = center[0]-radius-DieFaceSample.crop_padding
        cropX2 = center[0]+radius+DieFaceSample.crop_padding
        cropY1 = center[1]-radius-DieFaceSample.crop_padding
        cropY2 = center[1]+radius+DieFaceSample.crop_padding
        image_Crop_RGBA = cv.cvtColor(image_RGB[cropY1:cropY2,cropX1:cropX2,:],cv.COLOR_RGB2RGBA)
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.imshow(image_Crop_RGBA)
            plt.title("Cropped RGB")
            plt.show()
        image_Crop_RGBA[:,:,3] = mask[cropY1:cropY2,cropX1:cropX2]
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.imshow(image_Crop_RGBA[:,:,3],cmap='gray')
            plt.title("Cropped Alpha Mask")
            plt.show()
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(image_Crop_RGBA)
            plt.title("Cropped and Masked")
            plt.show()
        return image_Crop_RGBA
    
    @staticmethod
    def __crop_and_mask_template_image(image_RGB, log_level=LOG_LEVEL_FATAL):
    #1) Threshold saturation to get a grayscale image that rejects shadows and highlight the one die we expect near the middle
        image_HSV = cv.cvtColor(image_RGB,cv.COLOR_RGB2HSV)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(image_RGB)
            plt.title("Original Template Image")
            plt.show()
            plt.imshow(image_HSV[:,:,0],cmap='gray')
            plt.title("Hue")
            plt.show()
            plt.imshow(image_HSV[:,:,1],cmap='gray')
            plt.title("Saturation")
            plt.show()
            plt.imshow(image_HSV[:,:,2],cmap='gray')
            plt.title("Brightness")
            plt.show()

        image_Gray = image_HSV[:,:,1]
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(image_Gray,cmap='gray')
            plt.title("Grayscale Image (Saturation)")
            plt.show()
        _,image_BW = cv.threshold(image_Gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        #2) reject noise with an opening morphology (erode then dilate)
        image_Filtered_BW = image_BW.copy()
        image_Filtered_BW = cv.morphologyEx(image_Filtered_BW,cv.MORPH_OPEN,get_kernel(DieFaceSample.edge_opening_kernel_size),iterations=2)

        #3) find the biggest contour and create a convex hull around it. This is the die.
        contours, _ = cv.findContours(image_Filtered_BW,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(image_BW,cmap='gray')
            plt.title("Adaptive Threshold on Grayscale")
            plt.show()
            plt.imshow(image_Filtered_BW,cmap='gray')
            plt.title("Filtered Matching Areas")
            plt.show()
        if len(contours) < 1:
            if log_level >= LOG_LEVEL_ERROR:
                plt.imshow(image_Filtered_BW,cmap='gray')
                plt.show()
            raise Exception("Error cropping template image: No contour found")
        contour_Areas = [cv.contourArea(c) for c in contours]
        hull = cv.convexHull(contours[contour_Areas.index(max(contour_Areas))])
        image_Mask_BW = np.zeros(image_Gray.shape, dtype=np.uint8)
        cv.drawContours(image_Mask_BW, [hull], 0, 255,thickness=cv.FILLED)

        #4) circumscribe the convext hull to get the center and rotation-invariant "extents" of the die
        (x,y), r = cv.minEnclosingCircle(hull)
        center = (int(x),int(y))
        radius = int(r)
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.imshow(image_Mask_BW,cmap='gray')
            plt.title("Mask used to extract die from template")
            plt.show()
            image_ID = image_RGB.copy()
            cv.drawContours(image_ID,[hull],0,(255,255,255),3)
            cv.circle(image_ID,center,radius,(255,0,0),3)
            plt.imshow(image_ID)
            plt.title("Dice overlaid with contour and circumscribed boundary")
            plt.show()

        #5) crop around the circle (with padding) to get a scale-invariant image of the die. move the mask into the alpha channel
        
        return DieFaceSample.crop_sample_from_image(image_RGB, image_Mask_BW,center,radius,log_level)

    @staticmethod
    def __create_from_uncropped_image(image_or_file, geometry, log_level=LOG_LEVEL_FATAL):
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
        return DieFaceSample(DieFaceSample.__crop_and_mask_template_image(image_RGB, log_level), geometry, log_level)
    
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
    def create_from_image(image_or_file, geometry, autocrop=True, log_level=LOG_LEVEL_FATAL):
        if autocrop:
            return DieFaceSample.__create_from_uncropped_image(image_or_file, geometry, log_level)
        else:
            return DieFaceSample.__create_from_cropped_image(image_or_file, geometry, log_level)
        
    @staticmethod
    def create_from_image_files(image_files, geometry, autocrop=True, log_level=LOG_LEVEL_FATAL):
        if os.path.isdir(image_files):
            dir_files = [f for f in os.listdir(image_files) if os.path.isfile(os.path.join(image_files,f))]
            return [DieFaceSample.create_from_image(f, geometry,  autocrop, log_level) for f in dir_files]
        else:
            return [DieFaceSample.create_from_image(f, geometry,  autocrop, log_level) for f in image_files]

