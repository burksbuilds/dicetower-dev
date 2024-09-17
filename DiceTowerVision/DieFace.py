import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision.DieFaceSample import *

class DieFace:
    def __init__(self,die,value:int,template:DieFaceTemplate):
        self.die = die
        self.value = int(value)
        self.template = template
        # self.samples = list()

    # def add_samples(self, samples):

    #     for s in samples:
    #         self.samples.append(s)


    # def compare_to(self, other, camera_matrix=None, log_level=LOG_LEVEL_FATAL):
    #     results = [s.compare_to(other,camera_matrix,log_level) for s in self.samples]
    #     i = 0
    #     for r in results:
    #         r["sample_number"] = i
    #         r["face_value"] = self.value
    #         i += 1
    #     return results
    
    def compare_to_image(self, other, keypoints=None, descriptors=None, log_level=LOG_LEVEL_FATAL) -> DieFaceMatchResult:
        match_result = self.template.compare_to_image(other,keypoints,descriptors, log_level=log_level)
        match_result.face = self
        return match_result
    
    def compare_to_sample(self, sample, log_level=LOG_LEVEL_FATAL) -> DieFaceMatchResult:
        sample_image = sample.get_keypoint_detection_image(self.die.imaging_settings, log_level=log_level)
        sample_keypoints, sample_descriptors = ContourGroupMatcher.get_keypoints_and_descriptors(sample_image, log_level=log_level)
        return self.compare_to_image(sample_image, sample_keypoints, sample_descriptors, log_level=log_level)
    
    # @staticmethod
    # def create_from_image_file_template(image_file_template, value, geometry, imaging_settings=None, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
    #     samples = DieFaceSample.create_from_image_file_template(image_file_template, geometry, imaging_settings, autocrop, roi_mask, log_level)
    #     die_face = DieFace(value)
    #     die_face.add_samples(samples)
    #     return die_face
    


    # check that all samples within this face generally match
    # def compare_samples(self, log_level=LOG_LEVEL_FATAL):
    #     scores = list()
    #     for i in np.arange(0,len(self.samples)-1):
    #         for j in np.arange(i+1,len(self.samples)):
    #             result = self.samples[i].compare_to(self.samples[j])
    #             result_scores = get_scores_from_results([result])
    #             scores.append(result_scores[0])
    #     return scores
        
  
