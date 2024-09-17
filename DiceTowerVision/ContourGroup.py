import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from DiceTowerVision.DiceTowerTools import *

class ContourGroup:

    def __init__(self, image, contours=None, log_level=LOG_LEVEL_FATAL):
        self.base_image = image
        if contours is None:
            self.contours = self.find_contours(image, log_level=log_level)
        else:
            self.contours = contours
        self.mask = self.get_mask(self.contours, image.shape)
        self.image = cv.bitwise_and(self.mask, image)
        self.moments = cv.moments(self.image, binaryImage=True)
        self.area = self.moments["m00"]

        self.image_center = np.array([int(image.shape[1]//2), int(image.shape[0]//2)])
        if self.area > 0:
            self.centroid = np.array([int(self.moments["m10"]/self.area), int(self.moments["m01"]/self.area)])
        else:
            self.centroid = self.image_center
        centroid_from_center = self.centroid-self.image_center
        self.centroid_radius = np.linalg.norm(centroid_from_center)
        self.centroid_angle = np.degrees(np.arctan2(centroid_from_center[1],centroid_from_center[0]))
        
        if log_level >= LOG_LEVEL_VERBOSE:
            self.view()


    def view(self, title=None):
        plt.subplot(1,3,1)
        plt.imshow(self.base_image,cmap='gray')
        plt.title("Original")
        plt.subplot(1,3,2)
        plt.imshow(self.mask,cmap='gray')
        plt.title("Mask")
        plt.subplot(1,3,3)
        plt.imshow(self.image,cmap='gray')
        plt.plot([self.centroid[0],self.centroid[1],'ro'])
        plt.title("Final")
        plt.suptitle("Contour Group Feature")
        plt.show()

    def split(self, log_level=LOG_LEVEL_FATAL):
        return [ContourGroup(self.base_image,[c],log_level=log_level) for c in self.contours]
    
    def all_points(self):
        return np.concatenate(self.contours)
    
    def is_in_angle(self, start_angle=0, stop_angle=360):
        if start_angle < stop_angle:
            return self.centroid_angle > start_angle and self.centroid_angle < stop_angle
        else:
            return self.centroid_angle > start_angle or self.centroid_angle < stop_angle


    def distance_to(self, other):
        return np.linalg.norm(self.centroid-other.centroid)
    
    def get_elliptical_mask(self, margin=0, log_level=LOG_LEVEL_FATAL):
        points = np.ndarray((0,1,2),dtype=np.int32)
        for c in self.contours:
            points = np.append(points,c,axis=0)
        bounding_rect = cv.fitEllipse(points)
        mask = np.zeros(self.image.shape,dtype=np.uint8)
        mask = cv.ellipse(mask,box=bounding_rect, color=255, thickness=cv.FILLED)
        mask = cv.morphologyEx(mask,cv.MORPH_DILATE,get_kernel(3),iterations=margin)
        if log_level >= LOG_LEVEL_VERBOSE:
            image_RGB = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
            image_RGB[:,:,1] = cv.bitwise_or(image_RGB[:,:,1],mask)
            plt.imshow(image_RGB)
            plt.title("Elliptical Mask: Margin=%u"%(margin))
            plt.show()
        return mask


    @staticmethod
    def group_by_angle(groups, max_groups, distance, log_level=LOG_LEVEL_FATAL):
        if max_groups <= 0 or len(groups) == 0:
            return list()
        groups_remaining = [g for g in groups]
        combined_groups = list()
        for i in np.arange(0,max_groups):
            groups_remaining.sort(key= lambda g: -1*(g.area / max(distance,g.centroid_radius)))
            combined_groups.append(groups_remaining.pop(0))
            check_for_neighbors = True
            while check_for_neighbors and len(groups_remaining) > 0:
                check_for_neighbors = False
                for g in groups_remaining:
                    if combined_groups[i].distance_to(g) <= distance:
                        combined_groups[i] = ContourGroup.combine([combined_groups[i], g])
                        groups_remaining.remove(g)
                        check_for_neighbors = True
                        break
            if len(groups_remaining) == 0:
                break
        combined_groups.sort(key= lambda g: g.centroid_angle)
        return combined_groups


        
        #remove closest two groups, combine, and re-insert. repeat until max groups remaining
        # while len(groups_remaining) > max_groups:
        #     angles = np.array([g.centroid_angle for g in groups_remaining])
        #     angles_next = np.roll(angles,-1)
        #     angles_next[-1] += 360
        #     angle_diff = angles_next-angles
        #     i_min = np.argmin(angle_diff)
        #     i_min2 = np.mod(i_min+1,len(groups_remaining))
        #     combinedGroup = ContourGroup.combine([groups_remaining.pop(max(i_min,i_min2)),groups_remaining.pop(min(i_min,i_min2))],log_level=log_level)
        #     groups_remaining.append(combinedGroup)
        #     groups_remaining.sort(key=lambda g: g.centroid_angle)
        # return groups_remaining

    @staticmethod
    def group_by_distance(main_cg, available_cgs, distance, log_level=LOG_LEVEL_FATAL):
        remaining_cgs = list()
        for cg in available_cgs:
            if main_cg.distance_to(cg) <= distance:
                main_cg = ContourGroup.combine([main_cg,cg])
            else:
                remaining_cgs.append(cg)
        return main_cg, remaining_cgs
    
    @staticmethod
    def get_mask(contours, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if len(contours) > 0:
            mask = cv.drawContours(mask,contours,-1,color=255,thickness=cv.FILLED)
        return mask
    
   
    
    @staticmethod
    def find_contours(image, min_area=0.0, log_level=LOG_LEVEL_FATAL):
        contours, _ = cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        if min_area <= 0:
            return contours
        return np.array([contour for contour in contours if cv.contourArea(contour) >= min_area])
    
    @staticmethod
    def combine(contourgroups, log_level=LOG_LEVEL_FATAL):
        return ContourGroup(contourgroups[0].base_image,[c for cg in contourgroups for c in cg.contours],log_level=log_level)
