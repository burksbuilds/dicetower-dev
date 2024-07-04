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

    def compare_to(self, other, log_level=LOG_LEVEL_FATAL):
        results = [s.compare_to(other,log_level) for s in self.samples]
        for r in results:
            r["template_face"] = self
        return results
    


    
