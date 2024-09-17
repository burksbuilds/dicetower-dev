import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision import Die
from DiceTowerVision import DieFaceSample
import math
from itertools import compress

class DieSet:

    common_die_names = [4, 6, 8, 10, 12, 20, 100]

    def __init__(self):
        self.dies = dict()


    def view_die_color_space(self):
        for die_name, die in self.dies.items():
            thetas = np.deg2rad(2 * np.arange(die.imaging_settings.hue_start,(die.imaging_settings.hue_end if die.imaging_settings.hue_end > die.imaging_settings.hue_start else (die.imaging_settings.hue_end+180))+1,1))
            r_min = min(255-die.imaging_settings.number_brightness,die.imaging_settings.brightness_threshold)*np.ones(thetas.shape)
            r_max = max(255-die.imaging_settings.number_brightness,die.imaging_settings.brightness_threshold)*np.ones(thetas.shape)
            plt.polar(thetas, r_min, color=die.imaging_settings.hue_RGB)
            plt.polar(thetas, r_max, color=die.imaging_settings.hue_RGB)
            plt.fill_between(thetas, r_min, r_max, color=die.imaging_settings.hue_RGB)
            ax = plt.gca()
            ax.set_facecolor((0,0,0))
            ax.grid(False)
            print("%s: Hue [%u, %u] ; Brightness [%u, %u] ; num=%u, thresh=%u"%(die_name, die.imaging_settings.hue_start, die.imaging_settings.hue_end, r_min[0],r_max[0],die.imaging_settings.number_brightness,die.imaging_settings.brightness_threshold))
        plt.show()

    def add_die(self,name,die):
        if name in self.dies:
            raise KeyError()
        self.dies[name] = die

    def create_common_die_set_from_images(file_template, names=common_die_names, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        dies = DieSet()
        for name in names:
            die_template = file_template.replace("DR#",str(name))
            dies.add_die(str(name),Die.create_common_die_from_images(die_template,name,roi_mask=roi_mask, log_level=log_level))
        return dies
    
    def get_die_faces_from_image(self, image_RGB, rolls=None, roi_mask=None, rank_confidence=0.75, face_confidence=0.75, log_level=LOG_LEVEL_FATAL):
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(image_RGB)
            plt.show()
        image_HLS = cv.cvtColor(image_RGB, cv.COLOR_RGB2HLS)

        _, foreground_mask = cv.threshold(image_HLS[:,:,1],DieFaceSample.backround_brightness_threshold,255,cv.THRESH_BINARY)

        
        rolls_remaining = rolls
        if rolls_remaining is None:
            rolls_remaining = dict.fromkeys(self.dies.keys(),-1)
        die_location_mask = cv.copyTo(roi_mask,None)
        if die_location_mask is None:
            die_location_mask = 255*np.ones((image_RGB.shape[0], image_RGB.shape[1]),dtype=np.uint8)
        results = dict.fromkeys(self.dies.keys())
        for r in results.keys():
            results[r] = list()

        if log_level>=LOG_LEVEL_INFO:
            image_ID = np.copy(image_RGB)

        die_color_masks = dict()
        for die_name in rolls_remaining.keys():
            die_color_masks[die_name] = self.dies[die_name].get_color_mask_from_image(image_HLS, roi_mask,log_level=log_level)

        dies_to_match = [self.dies[die_name] for die_name in rolls_remaining.keys()]
        while True:
            die_matches = [m for m in self.__yield_potential_matches_from_image(dies_to_match,die_color_masks, die_location_mask,log_level)]
            die_confidence_scores = np.array([m.confidence for m in die_matches])
            for die_match in die_matches:
                if die_match.confidence < rank_confidence:
                    dies_to_match.remove(die_match.die)
            if log_level>=LOG_LEVEL_DEBUG:
                die_names = [m.die.name for m in die_matches]
                plt.bar( np.arange(0,len(die_confidence_scores)),die_confidence_scores, tick_label=die_names)
                plt.title("Confidence Scores for Selecting Which Die Type to Extract")
                plt.show()
            best_die_match = die_matches[np.argmax(die_confidence_scores)]

            if best_die_match.confidence <= rank_confidence:
                if log_level >= LOG_LEVEL_INFO:
                    print("Finished search because best die match, %s, has low confidence: %2f%%"%(best_die_match.die.name,100*best_die_match.confidence))
                break

            rolls_remaining[best_die_match.die.name] -= 1
            die_mask_radius = int(best_die_match.die.imaging_settings.average_circumscribed_radius*(1-(1-best_die_match.die.geometry.enscribed_perimeter_ratio)/2))

            sample = best_die_match.die.get_sample_at_point_from_image(image_RGB, best_die_match.point, image_HLS, mask=None, log_level=log_level)
            face_matches = best_die_match.die.compare_to_sample(sample, log_level=log_level)
            face_confidence_scores = np.array([m.confidence for m in face_matches])
            best_face_match = face_matches[np.argmax(face_confidence_scores)]
            passing_face_matches = list(compress(face_matches,face_confidence_scores >= face_confidence))

            if len(passing_face_matches) > 1: #multiple faces passed
                close_face_matches = list(compress(face_matches,np.bitwise_and( face_confidence_scores > (best_face_match.confidence-0.05) , face_confidence_scores < best_face_match.confidence)))
                
                if len(close_face_matches) > 0:
                    results[best_die_match.die.name].append((None,face_matches))
                    die_center_point = best_die_match.point
                    if log_level >= LOG_LEVEL_WARN:
                        print("WARNING: Multiple faces passed the confidence threshold: Found d%s with values %s"%(best_die_match.die.name,", ".join(["%u (c=%2.0f%%)"%(m.face.value, 100*m.confidence) for m in passing_face_matches])))
                        if log_level >= LOG_LEVEL_INFO:
                            best_die_match.die.view_match_results(passing_face_matches)
                            sample_image_BW = sample.get_keypoint_detection_image(best_die_match.die.imaging_settings)
                            for m in passing_face_matches:
                                m.view(sample_image_BW)
                else:
                    results[best_die_match.die.name].append((best_face_match,face_matches))
                    die_offset = get_sample_offset(sample, best_face_match.face.template, best_face_match.top_face_match.affine_warp)
                    die_center_point = (best_die_match.point[0]+int(die_offset[0]), best_die_match.point[1]+int(die_offset[1]))
            elif len(passing_face_matches) == 0: #no faces passed
                if log_level >= LOG_LEVEL_WARN:
                    print("WARNING: Face matching confidence too low: Found d%s at [%u,%u] with value %u (confidence=%2.0f%%)"%(best_die_match.die.name,best_die_match.point[0],best_die_match.point[1],best_face_match.face.value, 100*best_face_match.confidence))
                    if log_level >= LOG_LEVEL_INFO: #rerun with debug plots
                        best_face_match.view(sample.get_keypoint_detection_image(best_die_match.die.imaging_settings))
                        best_die_match.die.view_match_results(face_matches)
                        #_ = best_face_match.face.compare_to_sample(sample, max(log_level, LOG_LEVEL_DEBUG))
                results[best_die_match.die.name].append((None,face_matches))
                die_center_point = best_die_match.point
            else: #exactly one face passed
                if log_level>=LOG_LEVEL_DEBUG:
                    best_die_match.die.view_match_results(face_matches, "#%s"%best_die_match.die.name)
                # if log_level>= LOG_LEVEL_INFO and best_face_match.confidence < face_confidence * 1.1:
                #     print("INFO: Face matching confidence barely past threshold: Found d%s with value %u (confidence=%2.0f%%)"%(best_die_match.die.name,best_face_match.face.value, 100*best_face_match.confidence))
                #     best_face_match.view(sample.get_keypoint_detection_image(best_die_match.die.imaging_settings))
                close_face_matches = list(compress(face_matches,np.bitwise_and( face_confidence_scores > (best_face_match.confidence-0.05) , face_confidence_scores < best_face_match.confidence)))
                if len(close_face_matches) > 0:
                    if log_level >= LOG_LEVEL_WARN:
                        print("WARNING: other faces were very close to passing")
                        if log_level >= LOG_LEVEL_INFO:
                            sample_image_BW = sample.get_keypoint_detection_image(best_die_match.die.imaging_settings)
                            for m in close_face_matches:
                                m.view(sample_image_BW)
                results[best_die_match.die.name].append((best_face_match,face_matches,sample))
                die_offset = get_sample_offset(sample, best_face_match.face.template, best_face_match.top_face_match.affine_warp)
                #print(die_offset)
                die_center_point =  (best_die_match.point[0]+int(die_offset[0]), best_die_match.point[1]+int(die_offset[1]))
                # die_offset_distance = best_face_match #math.sqrt(match_results[best_result_index]["projected_offset_x"]**2 + match_results[best_result_index]["projected_offset_y"]**2)
                # die_offset_angle = math.atan2(match_results[best_result_index]["projected_offset_y"],match_results[best_result_index]["projected_offset_x"]) - math.radians(match_results[best_result_index]["rotation_angle"]) #might be negative of this
                # die_center_point = (int(die_center_point[0] + math.cos(die_offset_angle)*die_offset_distance), int(die_center_point[1] + math.sin(die_offset_angle)*die_offset_distance))
                

            die_location_mask = cv.circle(die_location_mask,die_center_point,die_mask_radius,0.0,cv.FILLED)
            if log_level>=LOG_LEVEL_INFO:
                if results[best_die_match.die.name][-1][0] is None: #best_face_match.confidence < face_confidence:
                    circle_color = (255,0,0)
                else:
                    circle_color = (0,255,0)
                cv.circle(image_ID,die_center_point,int(best_die_match.die.imaging_settings.average_circumscribed_radius*1.1),circle_color,10)
            if log_level >= LOG_LEVEL_DEBUG:
                print("Found d%s with value %u (score=%2f)"%(best_die_match.die.name,best_face_match.face.value,best_face_match.confidence))
                print(best_face_match.top_face_match)
                plt.imshow(die_location_mask,cmap='gray')
                plt.title("New Search Mask")
                plt.show()
            

        if log_level>=LOG_LEVEL_INFO:
            roll_total = 0.0
            
            textY = 150
            for die_name in results.keys():
                result_text = "D%s:"%die_name
                for die_result in results[die_name]:
                    if die_result[0] is not None:
                        result_text += " %u (%2.0f%%)"%(die_result[0].face.value,die_result[0].confidence*100)
                        if roll_total is not None:
                            roll_total += die_result[0].face.value
                    else:
                        result_text += " (?)"
                        roll_total = None
                cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),10)
                textY += 150
            result_text = "TOTAL: %s"%("?!" if roll_total is None else "%u"%roll_total)
            cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),10)
            
            plt.imshow(image_ID)
            plt.show()

        return results


    def __yield_potential_matches_from_image(self, dies, die_color_masks, die_location_mask, log_level=LOG_LEVEL_FATAL):
        for die in dies:
            result = die.get_best_point_from_mask(cv.bitwise_and(die_color_masks[die.name], die_location_mask), log_level=log_level)
            yield result
            

        


    
