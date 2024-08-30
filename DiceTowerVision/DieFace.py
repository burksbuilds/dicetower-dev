import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision.DieFaceSample import *

class DieFace:
    def __init__(self,value):
        self.value = int(value)
        self.samples = list()

    def add_samples(self, samples):

        for s in samples:
            self.samples.append(s)

    def compare_to(self, other, camera_matrix=None, log_level=LOG_LEVEL_FATAL):
        results = [s.compare_to(other,camera_matrix,log_level) for s in self.samples]
        i = 0
        for r in results:
            r["sample_number"] = i
            r["face_value"] = self.value
            i += 1
        return results
    
    @staticmethod
    def create_from_image_file_template(image_file_template, value, geometry, imaging_settings=None, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        samples = DieFaceSample.create_from_image_file_template(image_file_template, geometry, imaging_settings, autocrop, roi_mask, log_level)
        die_face = DieFace(value)
        die_face.add_samples(samples)
        return die_face

    # check that all samples within this face generally match
    def compare_samples(self, log_level=LOG_LEVEL_FATAL):
        scores = list()
        for i in np.arange(0,len(self.samples)-1):
            for j in np.arange(i+1,len(self.samples)):
                result = self.samples[i].compare_to(self.samples[j])
                result_scores = get_scores_from_results([result])
                scores.append(result_scores[0])
        return scores
        
    
