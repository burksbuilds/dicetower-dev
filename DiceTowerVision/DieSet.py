import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision import Die
import math

class DieSet:

    common_die_names = [4, 6, 8, 10, 12, 20, 100]

    def __init__(self):
        self.dies = dict()


    def view_die_color_space(self):
        for die_name, die in self.dies.items():
            thetas = np.deg2rad(2 * np.arange(die.imaging_settings["hue_start"],(die.imaging_settings["hue_end"] if die.imaging_settings["hue_end"] > die.imaging_settings["hue_start"] else (die.imaging_settings["hue_end"]+180))+1,1))
            r_min = min(255-die.imaging_settings["number_brightness"],die.imaging_settings["brightness_threshold"])*np.ones(thetas.shape)
            r_max = max(255-die.imaging_settings["number_brightness"],die.imaging_settings["brightness_threshold"])*np.ones(thetas.shape)
            plt.polar(thetas, r_min, color=die.imaging_settings["hue_RGB"])
            plt.polar(thetas, r_max, color=die.imaging_settings["hue_RGB"])
            plt.fill_between(thetas, r_min, r_max, color=die.imaging_settings["hue_RGB"])
            ax = plt.gca()
            ax.set_facecolor((0,0,0))
            ax.grid(False)
            print("%s: Hue [%u, %u] ; Brightness [%u, %u] ; num=%u, thresh=%u"%(die_name, die.imaging_settings["hue_start"], die.imaging_settings["hue_end"], r_min[0],r_max[0],die.imaging_settings["number_brightness"],die.imaging_settings["brightness_threshold"]))
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
    
    def get_die_faces_from_image(self, image_RGB, rolls=None, mask=None, rank_confidence=0.5, face_confidence=0.25, camera_matrix = None, scoring_function=get_scores_from_results, log_level=LOG_LEVEL_FATAL):
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(image_RGB)
            plt.show()
        image_HLS = cv.cvtColor(image_RGB, cv.COLOR_RGB2HLS)

        
        rolls_remaining = rolls
        if rolls_remaining is None:
            rolls_remaining = dict.fromkeys(self.dies.keys(),-1)
        die_location_mask = cv.copyTo(mask,None)
        if die_location_mask is None:
            die_location_mask = 255*np.ones((image_RGB.shape[0], image_RGB.shape[1]),dtype=np.uint8)
        results = dict.fromkeys(self.dies.keys())
        for r in results.keys():
            results[r] = list()

        if log_level>=LOG_LEVEL_INFO:
            image_ID = np.copy(image_RGB)

        die_color_masks = dict()
        for die_name in rolls_remaining.keys():
            die_color_masks[die_name] = self.dies[die_name].get_color_mask_from_image(image_HLS,log_level=log_level)


        while True:
            potential_matches = [m for m in self.__yield_potential_matches_from_image(rolls_remaining,die_color_masks, die_location_mask,log_level)]
            confidence_scores = np.array([m["confidence"] for m in potential_matches])
            if log_level>=LOG_LEVEL_DEBUG:
                plt.bar(np.arange(0,len(confidence_scores)),confidence_scores)
                plt.title("Confidence Scores for Selecting Which Die Type to Extract")
                plt.show()
            best_match = potential_matches[np.argmax(confidence_scores)]

            if best_match["confidence"] <= rank_confidence:
                if log_level >= LOG_LEVEL_INFO:
                    print("Finished search because best die match, %s, has low confidence: %f"%(best_match["die_name"],best_match["confidence"]))
                break
            

            die = self.dies[best_match["die_name"]]
            sample = die.get_sample_at_point_from_image(image_RGB, best_match["point"], image_HLS, die_color_masks[best_match["die_name"]], log_level=log_level)
            face_value, score, match_results, scores, best_result_index = die.get_best_face_match_from_sample(sample, camera_matrix, scoring_function=scoring_function, log_level=log_level)
            for r in match_results:
                r["die_name"] = best_match["die_name"]
                r["die_position_x"] = best_match["point"][0]
                r["die_position_y"] = best_match["point"][1]
                r["die_position_confidence"] = best_match["confidence"]

            die_mask_radius = int(die.imaging_settings["average_circumscribed_radius"]*(1-(1-die.geometry["enscribed_perimeter_ratio"])/2))
            die_center_point = best_match["point"]
            rolls_remaining[best_match["die_name"]] -= 1

            if score < face_confidence:
                if log_level >= LOG_LEVEL_WARN:
                    print("WARNING: Face matching confidence too low: Found d%s with value %u (score=%2f)"%(best_match["die_name"],face_value,score))
                    if log_level >= LOG_LEVEL_INFO: #rerun with debug plots
                        best_match_sample = die.faces[str(match_results[best_result_index]["face_value"])].samples[match_results[best_result_index]["sample_number"]]
                        sample.view_geometry()
                        sample.view_keypoints()
                        sample.view_moments()
                        _ = best_match_sample.compare_to(sample,camera_matrix,LOG_LEVEL_VERBOSE)
                        _ = scoring_function(match_results,max(log_level,LOG_LEVEL_DEBUG))
                results[best_match["die_name"]].append((None,score,match_results,scores,best_result_index))
            else:
                results[best_match["die_name"]].append((face_value,score,match_results,scores,best_result_index))
                die_offset_distance = math.sqrt(match_results[best_result_index]["projected_offset_x"]**2 + match_results[best_result_index]["projected_offset_y"]**2)
                die_offset_angle = math.atan2(match_results[best_result_index]["projected_offset_y"],match_results[best_result_index]["projected_offset_x"]) - math.radians(match_results[best_result_index]["rotation_angle"]) #might be negative of this
                die_center_point = (int(die_center_point[0] + math.cos(die_offset_angle)*die_offset_distance), int(die_center_point[1] + math.sin(die_offset_angle)*die_offset_distance))
            
            cv.circle(die_location_mask,die_center_point,die_mask_radius,0.0,cv.FILLED)
            if log_level>=LOG_LEVEL_INFO:
                if score < face_confidence:
                    cv.circle(image_ID,die_center_point,die.imaging_settings["average_circumscribed_radius"],(255,0,0),10)
                else:
                    cv.circle(image_ID,die_center_point,die.imaging_settings["average_circumscribed_radius"],(0,255,0),10)
            if log_level >= LOG_LEVEL_DEBUG:
                print("Found d%s with value %u (score=%2f)"%(best_match["die_name"],face_value,score))
                print(match_results[best_result_index])
                plt.imshow(die_location_mask,cmap='gray')
                plt.show()
            

        if log_level>=LOG_LEVEL_INFO:
            roll_total = 0.0
            
            textY = 150
            for die_name in results.keys():
                result_text = "D%s:"%die_name
                for die_result in results[die_name]:
                    if die_result[0] is not None:
                        result_text += " %u (%2.0f%%)"%(die_result[0],die_result[1]*100)
                        if roll_total is not None:
                            roll_total += die_result[0]
                    else:
                        result_text += " ?! (%2.0f%%)"%(die_result[1]*100)
                        roll_total = None
                cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),10)
                textY += 150
            result_text = "TOTAL: %s"%("?!" if roll_total is None else "%u"%roll_total)
            cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),10)
            
            plt.imshow(image_ID)
            plt.show()

        return results


    def __yield_potential_matches_from_image(self, rolls_remaining, die_color_masks, die_location_mask, log_level=LOG_LEVEL_FATAL):


        for die_name in rolls_remaining.keys():
            if rolls_remaining[die_name] != 0:
                die = self.dies[die_name]
                result = dict()
                result["die_name"] = die_name
                result["point"], result["confidence"] = die.get_best_point_from_masks(die_color_masks[die_name], die_location_mask, log_level=log_level)
                yield result
            

        


    
