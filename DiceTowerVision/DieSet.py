import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision import Die
from DiceTowerVision import DieFaceSample
import math
from itertools import compress
from DiceTowerVision.Die import DieSearchResult
from DiceTowerVision.DieFaceTemplate import DieFaceMatchResult

class DieMatchResult:

    def __init__(self, best_face_match, all_face_matches, best_die_match, sample):
        self.best_die_match = best_die_match
        self.best_face_match = best_face_match #might be none
        self.all_face_matches = all_face_matches
        self.sample = sample

    def get_center_point(self):
        if self.best_face_match is None:
            return self.best_die_match.point
        else:
            die_offset = get_sample_offset(self.sample, self.best_face_match.face.template, self.best_face_match.top_face_match.affine_warp)
            return (self.best_die_match.point[0]+int(die_offset[0]), self.best_die_match.point[1]+int(die_offset[1]))
        
    def get_flat_dicts(self):
        d = {"die_"+k : v for k,v in self.best_die_match.to_flat_dict().items()}
        argsort = np.argsort(np.array([-1*m.confidence for m in self.all_face_matches]))
        #rank = argsort[argsort[argsort]] #dont understand but it works: https://stackoverflow.com/questions/52199837/does-python-numpy-have-a-function-that-does-the-opposite-of-argsort-that-fills #worked for d6 but not d8...
        return [m.to_flat_dict(np.nonzero(argsort==i)[0][0]) | d for i,m in enumerate(self.all_face_matches)]
    
    @staticmethod
    def get_flat_dict_field_names(num_adjacent_faces):
        names = ["die_"+n for n in DieSearchResult.get_flat_dict_field_names()]
        names.extend(DieFaceMatchResult.get_flat_dict_field_names(num_adjacent_faces))
        return names
    



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

        #_, foreground_mask = cv.threshold(image_HLS[:,:,1],DieFaceSample.backround_brightness_threshold,255,cv.THRESH_BINARY)

        
        rolls_remaining = rolls
        if rolls_remaining is None:
            rolls_remaining = dict.fromkeys(self.dies.keys(),-1)
        die_location_mask = cv.copyTo(roi_mask,None)
        if die_location_mask is None:
            die_location_mask = 255*np.ones((image_RGB.shape[0], image_RGB.shape[1]),dtype=np.uint8)
        results = list()

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
                    results.append(DieMatchResult(None,face_matches,best_die_match,sample))
                    die_center_point = best_die_match.point
                    if log_level >= LOG_LEVEL_WARN:
                        print("WARNING: Multiple faces passed the confidence threshold: Found d%s with values %s"%(best_die_match.die.name,", ".join(["%u (c=%2.0f%%)"%(m.face.value, 100*m.confidence) for m in passing_face_matches])))
                        if log_level >= LOG_LEVEL_INFO:
                            best_die_match.die.view_match_results(passing_face_matches)
                            sample_image_BW = sample.get_keypoint_detection_image(best_die_match.die.imaging_settings)
                            for m in passing_face_matches:
                                m.view(sample_image_BW)
                else:
                    results.append(DieMatchResult(best_face_match,face_matches,best_die_match,sample))
                    die_offset = get_sample_offset(sample, best_face_match.face.template, best_face_match.top_face_match.affine_warp)
                    die_center_point = (best_die_match.point[0]+int(die_offset[0]), best_die_match.point[1]+int(die_offset[1]))
            elif len(passing_face_matches) == 0: #no faces passed
                if log_level >= LOG_LEVEL_WARN:
                    print("WARNING: Face matching confidence too low: Found d%s at [%u,%u] with value %u (confidence=%2.0f%%)"%(best_die_match.die.name,best_die_match.point[0],best_die_match.point[1],best_face_match.face.value, 100*best_face_match.confidence))
                    if log_level >= LOG_LEVEL_INFO: #rerun with debug plots
                        best_face_match.view(sample.get_keypoint_detection_image(best_die_match.die.imaging_settings))
                        best_die_match.die.view_match_results(face_matches)
                        #_ = best_face_match.face.compare_to_sample(sample, max(log_level, LOG_LEVEL_DEBUG))
                results.append(DieMatchResult(None,face_matches,best_die_match,sample))
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
                results.append(DieMatchResult(best_face_match,face_matches,best_die_match,sample))
                die_offset = get_sample_offset(sample, best_face_match.face.template, best_face_match.top_face_match.affine_warp)
                die_center_point =  (best_die_match.point[0]+int(die_offset[0]), best_die_match.point[1]+int(die_offset[1]))
                

            die_location_mask = cv.circle(die_location_mask,die_center_point,die_mask_radius,0.0,cv.FILLED)
            # if log_level>=LOG_LEVEL_INFO:
            #     if results[-1].best_face_match is None: #best_face_match.confidence < face_confidence:
            #         circle_color = (255,0,0)
            #     else:
            #         circle_color = (0,255,0)
            #     cv.circle(image_ID,die_center_point,int(best_die_match.die.imaging_settings.average_circumscribed_radius*1.1),circle_color,10)
            if log_level >= LOG_LEVEL_DEBUG:
                print("Found d%s with value %u (score=%2f)"%(best_die_match.die.name,best_face_match.face.value,best_face_match.confidence))
                print(best_face_match.top_face_match)
                plt.imshow(die_location_mask,cmap='gray')
                plt.title("New Search Mask")
                plt.show()
            

        if log_level>=LOG_LEVEL_INFO:
            # roll_total = 0.0
            # textY = 150
            # for die_name in results.keys():
            #     result_text = "D%s:"%die_name
            #     for die_result in results[die_name]:
            #         if die_result.best_face_match is not None:
            #             result_text += " %u (%2.0f%%)"%(die_result.best_face_match.face.value,die_result.best_face_match.confidence*100)
            #             if roll_total is not None:
            #                 roll_total += die_result.best_face_match.face.value
            #         else:
            #             result_text += " (?)"
            #             roll_total = None
            #     cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),10)
            #     textY += 150
            # result_text = "TOTAL: %s"%("?!" if roll_total is None else "%u"%roll_total)
            # cv.putText(image_ID,result_text,(50,textY),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),10)
            
            image_ID = self.overlay_results_on_image(image_ID, results, roi_mask=roi_mask, log_level=log_level)
            plt.imshow(image_ID)
            plt.axis(False)
            plt.show()

        return results


    def __yield_potential_matches_from_image(self, dies, die_color_masks, die_location_mask, log_level=LOG_LEVEL_FATAL):
        for die in dies:
            result = die.get_best_point_from_mask(cv.bitwise_and(die_color_masks[die.name], die_location_mask), log_level=log_level)
            yield result
            

    @staticmethod
    def calculate_total(results, minmax_die = None):
        if results is None:
            return None,None,None
        minmax_results = [r for r in results if minmax_die is not None and r.best_die_match.die.name == minmax_die]
        totaling_results = [r for r in results if minmax_die is None or r.best_die_match.die.name != minmax_die]
        total = np.sum(np.array([np.inf if r.best_face_match is None else r.best_face_match.face.value for r in totaling_results]))
        if minmax_die is not None and len(minmax_results) > 1 and np.all(np.array([r.best_face_match is not None for r in minmax_results])):
            min = total + np.min(np.array([np.inf if r.best_face_match is None else r.best_face_match.face.value for r in minmax_results]))
            max = total + np.max(np.array([np.inf if r.best_face_match is None else r.best_face_match.face.value for r in minmax_results]))
        else:
            min = None
            max = None
        total = total + np.sum(np.array([np.inf if r.best_face_match is None else r.best_face_match.face.value for r in minmax_results]))
        return total,min,max


    def overlay_results_on_image(self, image_RGB, results, minmax_die=None, roi_mask=None, draw_text=True, log_level=LOG_LEVEL_FATAL):
        image_out = np.zeros(image_RGB.shape, dtype=np.uint8)
        image_out = cv.copyTo(image_RGB,roi_mask)

        die_text = {k:list() for k in self.dies.keys()}

        if results is not None:
            for result in results:
                if result.best_face_match is None:
                    circle_color = (255,0,0)
                    text = "?"
                else:
                    circle_color = (0,255,0)
                    text = "%u"%result.best_face_match.face.value
                    if log_level >= LOG_LEVEL_INFO:
                        text += " (%u%%)"%(100*result.best_face_match.confidence)
                cv.circle(image_out,result.get_center_point(),int(result.best_die_match.die.imaging_settings.average_circumscribed_radius*1.1),circle_color,10)
                die_text[result.best_die_match.die.name].append(text)
        if not draw_text:
            return image_out
        
        label_x = 50
        data_x = 250
        text_y = 100
        text_y_step = 100
        for die_name,die_texts in die_text.items():
            cv.putText(image_out,die_name+":",(label_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)
            for die_text in die_texts:
                cv.putText(image_out,die_text,(data_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)
                if log_level >= LOG_LEVEL_INFO:
                    text_y += text_y_step
            if len(die_texts) == 0 or log_level < LOG_LEVEL_INFO:
                text_y += text_y_step
        total,min,max = self.calculate_total(results, minmax_die=minmax_die)
        
        cv.putText(image_out,"SUM:",(label_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)
        if total is not None:
            cv.putText(image_out,"?" if np.isinf(total) else str(total),(data_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)
        if min is not None and max is not None:
                text_y += text_y_step
                cv.putText(image_out,"MIN:",(label_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)
                cv.putText(image_out,"?" if np.isinf(min) else str(min),(data_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)
                text_y += text_y_step
                cv.putText(image_out,"MAX:",(label_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)
                cv.putText(image_out,"?" if np.isinf(max) else str(max),(data_x,text_y),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),8)


        return image_out
    
    def draw_gui(self, gui_image, scene_image, results, text_zone, scene_zone, sample_zone, scene_roi=None, message=None, log_level=LOG_LEVEL_FATAL):
        gui_image[:,:,:] = 0
        self.draw_scene(gui_image, scene_image, results, scene_zone, scene_roi=scene_roi, draw_circles=True)
        self.draw_result_samples(gui_image, scene_image, results, sample_zone)
        self.draw_result_text(gui_image,results,text_zone,message=message, border_padding=10,minmax_die='20')
        return

    def draw_result_text(self, gui_image, results, text_zone, message=None, border_padding=10, minmax_die='20'):
        total,min_v,max_v = self.calculate_total(results, minmax_die=minmax_die)

        die_text = {k:list() for k in self.dies.keys()}
        if results is not None:
            for result in results:
                if result.best_face_match is None:
                    text = "?"
                else:
                    text = "%u"%result.best_face_match.face.value
                die_text[result.best_die_match.die.name].append(text)

        num_lines = 5 + len(self.dies)
        line_spacing = int((text_zone[3]-2*border_padding)/num_lines)
        text_scale = cv.getFontScaleFromHeight(cv.FONT_HERSHEY_SIMPLEX,int(line_spacing*0.7),2) 
        max_scale = text_zone[2]/300 #max scale can fit ~15 charactgers in the text zone width
        if text_scale > max_scale:
            line_spacing = int(line_spacing * max_scale/text_scale)
            text_scale = max_scale
        vertical_padding = int(line_spacing*0.15)


        text_x = text_zone[0] + border_padding
        text_y = text_zone[1] + border_padding + line_spacing - vertical_padding
        widest_text = 0
        for die_name in die_text.keys():
            text_size,_ = cv.getTextSize(die_name+":",cv.FONT_HERSHEY_SIMPLEX,text_scale,2)
            widest_text = max(widest_text,text_size[0])
            cv.putText(gui_image,str(die_name)+":",(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)
            text_y += line_spacing
        text_y += line_spacing #extra gap before sum
        text_size,_ = cv.getTextSize("SUM:",cv.FONT_HERSHEY_SIMPLEX,text_scale,2)
        widest_text = max(widest_text,text_size[0])
        cv.putText(gui_image,"SUM:",(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)
        text_y += line_spacing
        text_size,_ = cv.getTextSize("MIN:",cv.FONT_HERSHEY_SIMPLEX,text_scale,2)
        widest_text = max(widest_text,text_size[0])
        if min_v is not None:
            cv.putText(gui_image,"MIN:",(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)
        text_y += line_spacing
        text_size,_ = cv.getTextSize("MAX:",cv.FONT_HERSHEY_SIMPLEX,text_scale,2)
        widest_text = max(widest_text,text_size[0])
        if max_v is not None:
            cv.putText(gui_image,"MAX:",(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)
        text_y += line_spacing
        if message is not None:
            cv.putText(gui_image,message,(text_x,text_y), cv.FONT_HERSHEY_SIMPLEX, text_scale,(255,255,255),2)

        text_x += widest_text + border_padding
        text_y = text_zone[1] + border_padding + line_spacing - vertical_padding
        for die_text_values in die_text.values():
            if len(die_text_values) > 0:
                cv.putText(gui_image,", ".join(die_text_values),(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)
            text_y += line_spacing
        text_y += line_spacing #extra gap before sum
        if total is not None:
            cv.putText(gui_image,"?" if np.isinf(total) else "%u"%total,(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)
        text_y += line_spacing
        if min_v is not None:
            cv.putText(gui_image,"?" if np.isinf(min_v) else "%u"%min_v,(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)
        text_y += line_spacing
        if max_v is not None:
            cv.putText(gui_image,"?" if np.isinf(max_v) else "%u"%max_v,(text_x,text_y),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),2)

    def draw_scene(self, gui_image, scene_image, results, scene_zone, scene_roi=None, draw_circles=True):
        if scene_image is None:
            return

        if results is not None and draw_circles:
            scene_copy = cv.copyTo(scene_image, None)
            for result in results:
                if result.best_face_match is None:
                    circle_color = (0,0,255)
                else:
                    circle_color = (0,255,0)
                cv.circle(scene_copy,result.get_center_point(),int(result.best_die_match.die.imaging_settings.average_circumscribed_radius*1.1),circle_color,10)
        else:
            scene_copy = scene_image

        if scene_roi is not None:
            scene_rect = cv.boundingRect(scene_roi)
        else:
            scene_rect = (0,0,scene_image.shape[0], scene_image.shape[1])

        resized_width = int(scene_rect[2] * scene_zone[3] / scene_rect[3]) #resize so the height fills the appropriate space in the zone
        scene_scaled = cv.resize(scene_image[scene_rect[1]:scene_rect[1]+scene_rect[3],scene_rect[0]:scene_rect[0]+scene_rect[2]],dsize=(resized_width,scene_zone[3]))
        gui_image[scene_zone[1]:scene_zone[1]+scene_zone[3],scene_zone[0]:scene_zone[0]+min(resized_width,scene_zone[2])] = scene_scaled[:,0:min(resized_width,scene_zone[2])]
        return


    def draw_result_samples(self, gui_image, scene_image, results, sample_zone):
        label_aspect_ratio = 8
        tile_aspect_ratio = 1 / (1 + 1/label_aspect_ratio) #image sample is always square
        if results is None or len(results) == 0:
            return
        grid_layout, tile_size, tiled_area = get_optimal_grid_layout((sample_zone[2], sample_zone[3]),tile_aspect_ratio,len(results))
        label_zone_height = tile_size[1]-tile_size[0]
        text_padding = int(0.2*label_zone_height)
        text_scale = cv.getFontScaleFromHeight(cv.FONT_HERSHEY_SIMPLEX,int(label_zone_height*0.6),1)
        r=0
        c=0
        for result in results:
            tile_x = sample_zone[0]+c*tile_size[0]+tiled_area[0]
            tile_y = sample_zone[1]+r*tile_size[1]+tiled_area[1]
            sample_image = DieFaceSample.crop_sample_from_image(scene_image,None,result.get_center_point(),result.best_die_match.die.imaging_settings.average_circumscribed_radius)
            if result.best_face_match is not None and result.best_face_match.top_face_match.affine_warp is not None:
                sample_image = cv.warpAffine(sample_image,result.best_face_match.top_face_match.affine_warp,sample_image.shape[0:2])
            gui_image[tile_y:tile_y+tile_size[0],tile_x:tile_x+tile_size[0]] = cv.resize(sample_image[:,:,0:3],dsize=(tile_size[0],tile_size[0]))
            label = result.best_die_match.die.name+": "+("?" if result.best_face_match is None else "%u (%u%%)"%(result.best_face_match.face.value, result.best_face_match.confidence*100))
            (tw, th), bl = cv.getTextSize(label,cv.FONT_HERSHEY_SIMPLEX,text_scale,1)
            cv.putText(gui_image,label,(tile_x+int((tile_size[0]-tw)/2),tile_y+tile_size[1]-text_padding),cv.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),1)
            c += 1
            if c >= grid_layout[1]:
                c = 0
                r += 1
        return

        
        


        



    
