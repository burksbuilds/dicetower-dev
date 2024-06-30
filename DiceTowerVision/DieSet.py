import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision import Die

class DieSet:

    common_die_ranks = [4, 6, 8, 10, 12, 20]

    def __init__(self):
        self.dies = dict()

    def add_die(self,name,die):
        if name in self.dies:
            raise KeyError()
        self.dies[name] = die

    def create_common_die_set_from_images(file_template, ranks=common_die_ranks, log_level=LOG_LEVEL_FATAL):
        dies = DieSet()
        for rank in ranks:
            die_template = file_template.replace("DR#",str(rank))
            dies.add_die(str(rank),Die.create_common_die_from_images(die_template,rank,log_level=log_level))
        return dies
    
    def get_die_faces_from_image(self, image_RGB, rolls=None, mask=None, confidence_threshold=0.25, log_level=LOG_LEVEL_FATAL):
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(image_RGB)
            plt.show()
        image_HSV = cv.cvtColor(image_RGB, cv.COLOR_RGB2HSV)

        
        rolls_remaining = rolls
        if rolls_remaining is None:
            rolls_remaining = dict.fromkeys(self.dies.keys(),-1)
        die_location_mask = mask
        if die_location_mask is None:
            die_location_mask = 255*np.ones((image_RGB.shape[0], image_RGB.shape[1]),dtype=np.uint8)
        results = dict.fromkeys(self.dies.keys())
        for r in results.keys():
            results[r] = list()

        if log_level>=LOG_LEVEL_INFO:
            image_ID = np.copy(image_RGB)

        while True:
            potential_matches = [m for m in self.yield_potential_matches_from_image(rolls_remaining,image_RGB,image_HSV, die_location_mask,log_level)]
            confidence_scores = np.array([m["confidence"] for m in potential_matches])
            if log_level>=LOG_LEVEL_DEBUG:
                plt.bar(np.arange(0,len(confidence_scores)),confidence_scores)
                plt.title("Confidence Scores for Selecting Which Die Type to Extract")
                plt.show()
            best_match = potential_matches[np.argmax(confidence_scores)]
            if best_match["confidence"] <= confidence_threshold:
                break
            die = self.dies[best_match["die_name"]]
            sample = die.get_sample_at_point_from_image(image_RGB, best_match["point"], image_HSV, log_level=log_level)
            face, _ = die.get_best_face_match_from_sample(sample, log_level=log_level)
            results[best_match["die_name"]].append(face)
            cv.circle(die_location_mask,best_match["point"],int(die.imaging_settings["average_circumscribed_radius"]*die.geometry["enscribed_perimeter_ratio"]),0.0,cv.FILLED)

            if log_level>=LOG_LEVEL_INFO:
                cv.circle(image_ID,best_match["point"],die.imaging_settings["average_circumscribed_radius"],(255,0,0),10)
                
            if log_level >= LOG_LEVEL_DEBUG:
                print("Found %s with value %u"%(best_match["die_name"],face.value))
                plt.imshow(die_location_mask,cmap='gray')
                plt.show()
        if log_level>=LOG_LEVEL_INFO:
            roll_total = 0
            
            textY = 150
            for die_name in results.keys():
                result_text = "D%s:"%die_name
                for die_result in results[die_name]:
                    result_text += " %u"%die_result.value
                    roll_total += die_result.value
                cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,5,(0,0,0),10)
                textY += 150
            result_text = "TOTAL: %u"%roll_total
            cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,5,(0,0,0),10)
            
            plt.imshow(image_ID)
            plt.show()

        return results


    def yield_potential_matches_from_image(self, rolls_remaining, image_RGB, image_HSV, mask, log_level=LOG_LEVEL_FATAL):
        for die_name in rolls_remaining.keys():
            if rolls_remaining[die_name] != 0:
                die = self.dies[die_name]
                result = dict()
                result["die_name"] = die_name
                result["point"], result["confidence"] = die.get_best_point_from_image(image_RGB, mask, image_HSV, log_level=log_level)
                yield result
            

        


    
