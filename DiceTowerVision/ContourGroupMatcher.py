import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from DiceTowerVision.DiceTowerTools import *
from dataclasses import dataclass, field
from itertools import compress

@dataclass
class AffineWarpComponents:
    angle:float = field(default=0) #degrees
    shear:float = field(default=0)
    scale_x:float = field(default=0)
    scale_y:float = field(default=0)

    @staticmethod
    def decompose_affine_warp(warp):
        result = AffineWarpComponents()
        angle = np.arctan2(warp[1,0],warp[1,1])
        result.angle = np.degrees(angle)
        result.scale_x = math.sqrt(warp[1,0]**2 + warp[1,1]**2)
        shear_skewy = warp[0,1]*math.cos(angle) + warp[1,1]*math.sin(angle)
        if abs(math.sin(angle)) > 0.0001:
            result.scale_y = (shear_skewy*math.cos(angle) - warp[0,1]) / math.sin(angle)
        else: 
            result.scale_y = (warp[1,1] - shear_skewy*math.sin(angle)) / math.cos(angle)
        if result.scale_y == 0:
            result.shear = np.inf
        else:
            result.shear = shear_skewy/result.scale_y
        # transformed_center = np.matmul(warp[:,0:2].transpose(), warp[:,2])
        # result.offset = transformed_center[0:2]
        return result
    

@dataclass
class ContourGroupSearchResult:
    matches_found:int = field(default=0)
    matches_good:int = field(default=0)
    matches_used:int = field(default=0)
    matches_used_list:list[object] = field(default=None)
    matches_used_keypoints:list[object] = field(default=None)
    affine_warp_components:AffineWarpComponents = field(default_factory= lambda : AffineWarpComponents())
    affine_warp:np.ndarray = field(default_factory= lambda: np.array([[1.0, 0.0, 0.0],[0.0,1.0,0.0]]))





class ContourGroupMatcher:

    __edge_distance_map_thickness = 3 #translates to a 'deadzone' when doing pixel comparisons of contour edges
    __area_comparison_erosion = 1
    __keypoint_detector = cv.ORB.create()
    __keypoint_matcher = cv.BFMatcher(normType=cv.NORM_HAMMING, crossCheck=True) #hamming is best paired with ORB features https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #__keypoint_matcher = cv.FlannBasedMatcher(dict(algorithm = 6,table_number = 6,key_size = 12,multi_probe_level = 1), dict(checks=100))
    __good_keypoint_match_distance_ratio = 0.75
    __mask_dilation = 10

    def __init__(self, group, log_level=LOG_LEVEL_FATAL):
        self.group = group
        self.keypoints, self.descriptors = self.get_keypoints_and_descriptors(group.image, log_level=log_level)
        self.hu_moments = cv.HuMoments(self.group.moments)
        self.all_contours, _ = cv.findContours(self.group.image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self.all_contour_len = sum([len(c) for c in self.all_contours])
        self.edge_distance_map = self.get_edge_distance_map(self.group.image, self.all_contours, self.__edge_distance_map_thickness, log_level=log_level)
        self.mask = cv.morphologyEx(self.group.image, cv.MORPH_DILATE, get_kernel(3), iterations=self.__mask_dilation)

    @staticmethod
    def get_edge_distance_map(image, contours, thickness, log_level=LOG_LEVEL_FATAL):
        contour_image = 255*np.ones(image.shape, dtype=np.uint8)
        contour_image = cv.drawContours(contour_image, contours, -1, 0, thickness=thickness)
        edge_distance_map = cv.distanceTransform(contour_image, cv.DIST_L2,5)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.subplot(1,2,1)
            plt.imshow(contour_image, cmap='gray')
            plt.subplot(1,2,2)
            edge_distance_map_norm = edge_distance_map.astype(np.float32) / (image.shape[0]+image.shape[1])
            plt.imshow(edge_distance_map_norm, cmap='gray')
            plt.suptitle("Contour Matching Map")
            plt.show()
        return edge_distance_map.astype(np.uint8)

    def find_in_image(self,image, mask=None, keypoints=None, descriptors=None, allow_distortion=False, log_level=LOG_LEVEL_FATAL):
        result = ContourGroupSearchResult()
        if log_level >= LOG_LEVEL_VERBOSE and mask is not None:
            image_debug = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            image_debug[:,:,0] = image
            image_debug[:,:,1] = mask
            plt.imshow(image_debug)
            plt.title("Image to Search and Mask")
            plt.show()
        if keypoints is None or descriptors is None:
            keypoints, descriptors = self.__keypoint_detector.detectAndCompute(image, mask=mask)
            
        all_matches = self.__keypoint_matcher.match(descriptors, self.descriptors)
        good_matches = all_matches
        #all_matches = self.__keypoint_matcher.knnMatch(descriptors, self.descriptors, k=2)
        #good_matches = [m[0] for m in all_matches if len(m) ==2 and m[0].distance < self.__good_keypoint_match_distance_ratio*m[1].distance]  # Ratio test only matters if you use knnMatch:
        
        result.matches_found = len(all_matches)
        result.matches_good = len(good_matches)


        
        if len(good_matches) < 3:
            if log_level>=LOG_LEVEL_VERBOSE:
                comparison = cv.drawMatches(image,keypoints,self.group.image,self.keypoints,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(comparison)
                plt.title("Good Matches")
                plt.show()
            return result
        
        other_points = np.float32([ keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        self_points = np.float32([ self.keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        if allow_distortion:
            warp, inliers = cv.estimateAffine2D(other_points, self_points)
        else:
            warp, inliers = cv.estimateAffinePartial2D(other_points, self_points)
        result.affine_warp = warp
        result.matches_used = len(inliers)
        result.matches_used_list = list(compress(good_matches,inliers))
        result.matches_used_keypoints = keypoints
        if warp is None or result.matches_used == 0:
            return result

        result.affine_warp_components = AffineWarpComponents.decompose_affine_warp(warp)

        
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.subplot(1,2,1)
            comparison = cv.drawMatches(image,keypoints,self.group.image,self.keypoints,list(compress(good_matches,inliers)), None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(comparison)
            plt.title("Matches Used")
            plt.suptitle("Angle: %f ; Scale: (%f,%f) ; Shear: %f"%(result.affine_warp_components.angle,result.affine_warp_components.scale_x,result.affine_warp_components.scale_y,result.affine_warp_components.shear))
            corrected_image = cv.warpAffine(image, warp, (self.group.image.shape[1],self.group.image.shape[0]))
            plt.subplot(1,2,2)
            plt.imshow(corrected_image)
            plt.title("Corrected Image")
            plt.gcf().set_size_inches(8,4)
            plt.show()

        return result
        

    def compare_contour_area(self, image, mask=None, log_level=LOG_LEVEL_FATAL):
        other_image = image
        if mask is not None:
            other_image = cv.bitwise_and(mask,other_image)
        #other_image = cv.bitwise_and(other_image,self.mask)
        
        image_diff = cv.bitwise_xor(other_image, self.group.image)
        image_diff_weighted = cv.bitwise_and(image_diff,self.edge_distance_map)
        image_diff_eroded = cv.morphologyEx(image_diff, cv.MORPH_ERODE, get_kernel(3), iterations=self.__area_comparison_erosion)
        if self.all_contour_len > 0:
            error = np.count_nonzero(image_diff) / self.all_contour_len
            error_eroded = np.count_nonzero(image_diff_eroded) / self.all_contour_len
        else:
            error = np.inf
            error_eroded = np.inf

        if log_level>=LOG_LEVEL_DEBUG:
            diff_RGB = np.zeros((self.group.image.shape[1], self.group.image.shape[0],3), np.uint8)
            diff_RGB[:,:,1] = self.group.image
            diff_RGB[:,:,2] = other_image
            plt.subplot(1,3,1)
            plt.imshow(diff_RGB)
            plt.title("Overlay")
            plt.subplot(1,3,2)
            plt.imshow(image_diff,cmap='gray')
            plt.title("Original Error: %f"%(error))
            plt.subplot(1,3,3)
            plt.imshow(image_diff_eroded,cmap='gray')
            plt.title("Eroded n=%u Error: %f"%(self.__area_comparison_erosion,error_eroded))
            plt.suptitle("Overlay of Template and Sample Images")
            plt.gcf().set_size_inches(8,4)
            plt.show()
        return error,error_eroded

    @staticmethod
    def get_keypoints_and_descriptors(image, mask=None, log_level=LOG_LEVEL_FATAL):
        keypoints, descriptors = ContourGroupMatcher.__keypoint_detector.detectAndCompute(image, mask)
        if log_level >= LOG_LEVEL_VERBOSE:
            ContourGroupMatcher.view_keypoints(image, keypoints)
        return keypoints, descriptors

    @staticmethod
    def view_keypoints(image, keypoints):
        image_debug = cv.copyTo(image,None)
        image_debug = cv.drawKeypoints(image_debug,keypoints,0,(0,0,255),flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        plt.subplot(1,2,1)
        plt.imshow(image,cmap='gray')
        plt.title("Feature")
        plt.subplot(1,2,2)
        plt.imshow(image_debug,cmap='gray')
        plt.title("Keypoints")
        plt.show()


    # def compare_contour_edge(self, image, mask, log_level=LOG_LEVEL_FATAL):
    #     other_image = image
    #     if mask is not None:
    #         other_image = cv.bitwise_and(mask,other_image)
    #     _, other_contours = cv.findContours(other_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    #     other_contour_len = sum([len(c) for c in other_contours])
    #     other_edge_distance_map = self.get_edge_distance_map(other_image,other_contours,self.__edge_distance_map_thickness,log_level=log_level)
    #     other_error_from_template = self.get_contour_error(other_contours, self.edge_distance_map) / other_contour_len
    #     template_error_from_other = self.get_contour_error(self.all_contours, other_edge_distance_map) / self.all_contour_len
    #     return other_error_from_template + template_error_from_other

    # @staticmethod
    # def get_contour_error(contours,map):
    #     error = 0
    #     for contour in contours:
    #         for point in contour:
    #             error += map[point[1],point[0]]
    #     return error


