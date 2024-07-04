import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision import DieFace
from DiceTowerVision import DieFaceSample
from collections import namedtuple
import math


class Die:
    geometry_keys = ["enscribed_perimeter_ratio", "circumscribed_face_ratio", "enscribed_face_ratio", "perimeter_edges", "face_edges"]
    imaging_settings_keys = ["hue_start","hue_end","saturation_threshold","average_circumscribed_radius"]

    __hue_filter_capture = 0.95

    def __init__(self, face_values, geometry=None, imaging_settings=None, log_level=LOG_LEVEL_FATAL):
        self.faces = {str(v):DieFace(v) for v in face_values}
        self.geometry = geometry
        if self.geometry is None:
            self.geometry = Die.get_common_die_geometry(len(face_values))
        self.imaging_settings = imaging_settings
        if self.imaging_settings is None:
            self.imaging_settings = self.calculate_imaging_settings(log_level)

    @staticmethod
    def create_from_images(file_template, face_values, geometry=None, imaging_settings=None, autocrop=True, log_level=LOG_LEVEL_FATAL):
        die = Die(face_values, geometry, imaging_settings,log_level)
        for fv in die.faces.keys():
            image_path = file_template.replace("FV#",str(fv))
            die.faces[fv].add_samples([DieFaceSample.create_from_image(image_path,die.geometry,autocrop,log_level)])
        if imaging_settings is None:
            die.imaging_settings = die.calculate_imaging_settings(log_level)
        return die
    
    @staticmethod
    def create_common_die_from_images(file_template, rank, log_level=LOG_LEVEL_FATAL):
        return Die.create_from_images(file_template,Die.get_common_die_face_values(rank),Die.get_common_die_geometry(rank),log_level=log_level)
    


    def compare_to(self, other, log_level=LOG_LEVEL_FATAL):
        results = [self.faces[face_name].compare_to(other,log_level) for face_name in self.faces]
        return [sub_result for result in results for sub_result in result]
    



    def calculate_imaging_settings(self, log_level=LOG_LEVEL_FATAL):
        imaging_settings = dict.fromkeys(Die.imaging_settings_keys,0)

        sample_count = 0
        hues = np.zeros((180,1),dtype=np.float32)
        saturations = np.zeros((256,1),dtype=np.float32)
        saturations_inv = np.zeros((256,1),dtype=np.float32)
        average_circumscribed_radius = 0.0
        for face_name in self.faces.keys():
            for sample in self.faces[face_name].samples:
                sample_count += 1
                average_circumscribed_radius += sample.circumscribed_radius
                image_hsv = cv.cvtColor(cv.cvtColor(sample.image,cv.COLOR_RGBA2RGB),cv.COLOR_RGB2HSV)
                _,saturation_mask = cv.threshold(image_hsv[:,:,1],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                hues = np.add(hues,cv.calcHist([image_hsv[:,:,0]],[0],saturation_mask,[180],[0,180]))
                _,saturation_mask_inv = cv.threshold(image_hsv[:,:,1],0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                saturations = np.add(saturations,cv.calcHist([image_hsv[:,:,1]],[0],saturation_mask,[256],[0,256]))
                saturations_inv = np.add(saturations_inv, cv.calcHist([image_hsv[:,:,1]],[0],saturation_mask_inv,[256],[0,256]))
        if sample_count == 0:
            imaging_settings["hue_start"] = 0
            imaging_settings["hue_end"] = 179
            imaging_settings["saturation_threshold"] = 0
            imaging_settings["average_circumscribed_radius"] = 0
            return imaging_settings

        hues = hues / np.sum(hues)
        saturations = saturations / np.sum(saturations)
        saturations_inv = saturations_inv / np.sum(saturations_inv)
        if log_level >= LOG_LEVEL_DEBUG:
            plt.plot(hues)
            plt.title("Hue Histogram for all Die Face Samples")
            plt.show()
            plt.plot(saturations)
            plt.plot(saturations_inv)
            plt.title("Saturation Histogram for all Die Face Samples")
            plt.show()

        hue_sums = np.copy(hues)
        shift_count=1
        while np.max(hue_sums) < self.__hue_filter_capture:
            hue_sums = np.add(hue_sums,np.roll(hues,-1*shift_count))
            shift_count += 1

        imaging_settings["hue_start"] = int(np.argmax(hue_sums))
        imaging_settings["hue_end"] = (imaging_settings["hue_start"] + shift_count)%180
        imaging_settings["saturation_threshold"] = int((min(np.nonzero(saturations)[0]) + max(np.nonzero(saturations_inv)[0]))/2)
        imaging_settings["average_circumscribed_radius"] = int(average_circumscribed_radius / sample_count)
        return imaging_settings


        
    def get_color_mask_from_image(self, image_HSV, log_level=LOG_LEVEL_FATAL):
        if self.imaging_settings["hue_start"] > self.imaging_settings["hue_end"]:
            mask = cv.inRange(image_HSV,(0,self.imaging_settings["saturation_threshold"],0),(self.imaging_settings["hue_end"],255,255))
            mask = np.add(mask,cv.inRange(image_HSV,(self.imaging_settings["hue_start"],self.imaging_settings["saturation_threshold"],0),(180,255,255)))
        else:
            mask = cv.inRange(image_HSV,(self.imaging_settings["hue_start"],self.imaging_settings["saturation_threshold"],0),(self.imaging_settings["hue_end"],255,255))
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(mask, cmap='gray')
            plt.title("Color Filter Between Hue=[%u, %u] with Saturation > %u"%(self.imaging_settings["hue_start"],self.imaging_settings["hue_end"],self.imaging_settings["saturation_threshold"]))
            plt.show() 
        return fill_all_contours(mask,log_level)
    
    #confidence should be 1 when max dist_transform is the same as enscribed periemter
    def get_best_point_from_image(self, image_RGB, mask=None, image_HSV=None, log_level=LOG_LEVEL_FATAL):
        if image_HSV is None:
            image_HSV = cv.cvtColor(image_RGB,cv.COLOR_RGB2HSV)
        color_mask = self.get_color_mask_from_image(image_HSV,log_level)
        die_mask = color_mask
        if mask is not None:
            die_mask = cv.bitwise_and(die_mask,mask)
        dist_transform = cv.distanceTransform(die_mask,cv.DIST_L2,5)
        _, maxVal, _, maxP = cv.minMaxLoc(dist_transform)
        confidence = maxVal / (self.imaging_settings["average_circumscribed_radius"]*self.geometry["enscribed_perimeter_ratio"])
        if log_level >= LOG_LEVEL_DEBUG:
            plt.imshow(dist_transform,cmap='gray')
            plt.title("Distance Transform, max=%f of Enscribed Perimeter"%(confidence))
            plt.show()
        return maxP, confidence
    
    def get_sample_at_point_from_image(self, image_RGB, point, image_HSV=None, log_level=LOG_LEVEL_FATAL):
        if image_HSV is None:
            image_HSV = cv.cvtColor(image_RGB,cv.COLOR_RGB2HSV)
        color_mask = self.get_color_mask_from_image(image_HSV,log_level)
        return DieFaceSample(DieFaceSample.crop_sample_from_image(image_RGB, color_mask, point, self.imaging_settings["average_circumscribed_radius"],log_level),self.geometry,log_level)
    

        
    def get_best_face_match_from_sample(self, sample, log_level=LOG_LEVEL_FATAL):
        results = self.compare_to(sample,log_level)
        good_keypoint_scores = get_scores_from_good_keypoints(results,log_level)
        used_keypoint_scores = get_scores_from_used_keypoints(results,log_level)
        weighted_scores = get_weighted_scores([good_keypoint_scores,used_keypoint_scores],[1,1],log_level)
        best_index = np.argmax(weighted_scores)
        return results[best_index]["template_face"], weighted_scores[best_index]


    @staticmethod
    def get_common_die_geometry(rank):
        g = dict.fromkeys(Die.geometry_keys)
        match rank:
            case 4:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/3)
                g["circumscribed_face_ratio"] = 1
                g["enscribed_face_ratio"] = math.cos(math.pi/3)
                g["perimeter_edges"] = 3
                g["face_edges"] = 3
            case 6:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/4)
                g["circumscribed_face_ratio"] = 1.0
                g["enscribed_face_ratio"] = math.cos(math.pi/4)
                g["perimeter_edges"] = 4
                g["face_edges"] = 4
            case 8:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/6)
                g["circumscribed_face_ratio"] = 1.0
                g["enscribed_face_ratio"] = math.cos(math.pi/3)
                g["perimeter_edges"] = 6
                g["face_edges"] = 3
            case 10:
                g["enscribed_perimeter_ratio"] = 0.747
                g["circumscribed_face_ratio"] = 0.725
                g["enscribed_face_ratio"] = 0.365
                g["perimeter_edges"] = 6
                g["face_edges"] = 4
            case 12:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/10)
                g["circumscribed_face_ratio"] = 0.618
                g["enscribed_face_ratio"] = 0.500
                g["perimeter_edges"] = 10
                g["face_edges"] = 5
            case 20:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/6)
                g["circumscribed_face_ratio"] = 0.619
                g["enscribed_face_ratio"] = 0.313
                g["perimeter_edges"] = 6
                g["face_edges"] = 3
            case _:
                raise ValueError("Unsupported Die Rank for Default Geometry")
        return g

    @staticmethod
    def get_common_die_face_values(rank, start=1, step=1):
        return np.arange(start,rank+start,step)

