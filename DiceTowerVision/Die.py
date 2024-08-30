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
    geometry_keys = ["enscribed_perimeter_ratio", "circumscribed_face_ratio", "enscribed_face_ratio", "pixel_comparison_ratio", "perimeter_edges", "face_edges"]
    imaging_settings_keys = ["hue_start","hue_end","average_circumscribed_radius","number_brightness","brightness_threshold"]

    __hue_filter_capture = 0.90
    __hue_filter_capture_adj = 0.025
    #__hue_saturation_threshold = 150
    __hue_brightness_lower_threshold = 50
    __hue_brightness_upper_threshold = 205
    __hue_brightness_lower_limit = 25
    __hue_brightness_upper_limit = 230
    __number_test_threshold = 50
    __backround_brightness_threshold = 25
    # __white_number_threshold = 175
    __number_test_capture = 0.05
    __brightness_gradient_indicator = 0.01
    #__white_die_threshold = 150
    #__white_die_capture = 0.25
    #__hue_saturation_capture = 0.75
    #__number_saturation_capture = 0.05
    __hue_buffer = 5

    def __init__(self, face_values, geometry=None, imaging_settings=None, log_level=LOG_LEVEL_FATAL):
        self.faces = {str(v):DieFace(v) for v in face_values}
        self.geometry = geometry
        if self.geometry is None:
            self.geometry = Die.get_common_die_geometry(len(face_values))
        self.imaging_settings = imaging_settings
        if self.imaging_settings is None:
            self.imaging_settings = self.get_default_imaging_settings()

    @staticmethod
    def create_from_images(file_template, face_values, geometry=None, imaging_settings=None, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        die = Die(face_values, geometry, imaging_settings,log_level)
        fid = 0
        for fv in die.faces.keys():
            fid += 1
            image_path = file_template.replace("FV#",str(fid))
            die.faces[fv].add_samples(DieFaceSample.create_from_image_file_template(image_path,die.geometry,imaging_settings,autocrop, roi_mask=roi_mask, log_level=log_level))
        if imaging_settings is None:
            die.imaging_settings = die.calculate_imaging_settings(log_level)
            die.update_faces(log_level)
        return die
    
    @staticmethod
    def create_common_die_from_images(file_template, name, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        die_info = Die.get_common_die_info(name)
        return Die.create_from_images(file_template,Die.get_common_die_face_values(die_info[0],die_info[1],die_info[2]),Die.get_common_die_geometry(die_info[0]) ,roi_mask=roi_mask, log_level=log_level)
    


    def compare_to(self, other, camera_matrix=None, log_level=LOG_LEVEL_FATAL):
        if log_level >= LOG_LEVEL_DEBUG:
            other.view_geometry()
            other.view_keypoints()
            other.view_moments()
        results = [self.faces[face_name].compare_to(other,camera_matrix,log_level) for face_name in self.faces]
        return [sub_result for result in results for sub_result in result]
    
    def update_faces(self, log_level=LOG_LEVEL_FATAL):
        for face_name in self.faces.keys():
            for sample in self.faces[face_name].samples:
                sample.update_keypoints(self.imaging_settings, log_level)
                sample.update_moments(log_level)


    @staticmethod
    def get_default_imaging_settings(sample=None):
        imaging_settings = dict.fromkeys(Die.imaging_settings_keys,0)

        if sample is None:
            imaging_settings["hue_start"] = 0
            imaging_settings["hue_max"] = 90
            imaging_settings["hue_end"] = 179
            imaging_settings["hue_RGB"] = (255,255,255)
            imaging_settings["average_circumscribed_radius"] = 0
            imaging_settings["number_brightness"] = 0
            imaging_settings["brightness_threshold"] = 0
            return imaging_settings

        test_die = Die([0])
        test_die.faces["0"].add_samples(sample)
        return test_die.calculate_imaging_settings()

    

    def calculate_imaging_settings(self, log_level=LOG_LEVEL_FATAL):
        imaging_settings = dict.fromkeys(Die.imaging_settings_keys,0)

        sample_count = 0
        brightnesses = np.zeros((256,1),dtype=np.float32)
        average_circumscribed_radius = 0

        #determine white vs black numbers
        for face_name in self.faces.keys():
            for sample in self.faces[face_name].samples:
                sample_count += 1
                average_circumscribed_radius += sample.circumscribed_radius
                #image_hls = cv.cvtColor(cv.cvtColor(sample.image,cv.COLOR_RGBA2RGB),cv.COLOR_RGB2HLS)
                sample_mask = sample.image[:,:,3] #alpha channel is a mask of the foreground
                brightnesses = np.add(brightnesses,cv.calcHist([sample.image_hls[:,:,1]],[0], sample_mask,[256],[0,256]))

        if sample_count == 0:
            imaging_settings["hue_start"] = 0
            imaging_settings["hue_max"] = 90
            imaging_settings["hue_end"] = 179
            imaging_settings["hue_RGB"] = (255,255,255)
            imaging_settings["average_circumscribed_radius"] = 0
            imaging_settings["number_brightness"] = 0
            imaging_settings["brightness_threshold"] = 0
            return imaging_settings

        imaging_settings["average_circumscribed_radius"] = int(average_circumscribed_radius / sample_count)
        brightnesses = brightnesses / np.sum(brightnesses)
        brightnesses_cum = np.cumsum(brightnesses)
        if brightnesses_cum[self.__number_test_threshold] > self.__number_test_capture: #black numbers
            imaging_settings["number_brightness"] = 0
        else:
            imaging_settings["number_brightness"] = 255


        hues = np.zeros((180,1),dtype=np.float32)

        #determine hue of die
        for face_name in self.faces.keys():
            for sample in self.faces[face_name].samples:
                sample_count += 1
                average_circumscribed_radius += sample.circumscribed_radius
                #image_hls = cv.cvtColor(cv.cvtColor(sample.image,cv.COLOR_RGBA2RGB),cv.COLOR_RGB2HLS)
                sample_mask = sample.image[:,:,3] #alpha channel is a mask of the foreground
                body_brightness_mask = cv.inRange(sample.image_hls[:,:,1],min(self.__hue_brightness_lower_threshold,255-imaging_settings["number_brightness"]),max(self.__hue_brightness_upper_threshold,255-imaging_settings["number_brightness"]))
                #_, saturation_mask = cv.threshold(image_hls[:,:,2],self.__hue_saturation_threshold,255,cv.THRESH_BINARY)
                hue_mask = cv.bitwise_and(sample_mask,body_brightness_mask)
                hue_mask = cv.morphologyEx(hue_mask,cv.MORPH_ERODE,get_kernel(3),iterations=2) #edges might have weird colors, white, or black
                hues = np.add(hues,cv.calcHist([sample.image_hls[:,:,0]],[0],hue_mask,[180],[0,180]))
                if log_level >= LOG_LEVEL_VERBOSE:
                    plt.subplot(2,3,1)
                    plt.imshow(sample.image)
                    plt.title("RGBA Image")
                    plt.subplot(2,3,2)
                    plt.imshow(sample.image_hls[:,:,0],cmap='gray')
                    plt.title("Image Hue")
                    plt.subplot(2,3,3)
                    plt.imshow(sample.image_hls[:,:,1],cmap='gray')
                    plt.title("Image Brightness")
                    plt.subplot(2,3,4)
                    plt.imshow(sample_mask,cmap='gray')
                    plt.title("Mask for Sample ROI")
                    plt.subplot(2,3,5)
                    plt.imshow(body_brightness_mask,cmap='gray')
                    plt.title("Mask for Number Rejection")
                    plt.subplot(2,3,6)
                    plt.imshow(hue_mask,cmap='gray')
                    plt.title("Mask for Hue Detection")
                    plt.show()


        
        hues = hues / np.sum(hues)
        hue_sums = np.copy(hues)
        shift_count=1
        while np.max(hue_sums) < (self.__hue_filter_capture): # - self.__hue_filter_capture_adj*self.geometry["adjacent_faces"] ):
            hue_sums = np.add(hue_sums,np.roll(hues,-1*shift_count))
            shift_count += 1


        imaging_settings["hue_start"] = int(np.argmax(hue_sums)-self.__hue_buffer)%180
        imaging_settings["hue_end"] = (imaging_settings["hue_start"] + shift_count + 2*self.__hue_buffer)%180
        imaging_settings["hue_max"] = int(np.argmax(hues))
        test_pixel = 255*np.ones((1,1,3), dtype=np.uint8)
        test_pixel[0,0,0] = imaging_settings["hue_max"]
        test_pixel_RGB = cv.cvtColor(test_pixel,cv.COLOR_HSV2RGB).astype(np.float32)/255
        imaging_settings["hue_RGB"] = (test_pixel_RGB[0,0,0],test_pixel_RGB[0,0,1],test_pixel_RGB[0,0,2])

        
        if log_level >= LOG_LEVEL_DEBUG:
            plt.plot(hues)
            plt.title("Hue Histogram: [%u, %u]"%(imaging_settings["hue_start"],imaging_settings["hue_end"]))
            plt.show()

        #determine brightness cutoff using hue as a more accurate mask than the initial brightness estimate
        
        body_brightnesses = np.zeros((256,1),dtype=np.float32)
        number_brightnesses = np.zeros((256,1),dtype=np.float32)
        for face_name in self.faces.keys():
            for sample in self.faces[face_name].samples:
                #image_hls = cv.cvtColor(cv.cvtColor(sample.image,cv.COLOR_RGBA2RGB),cv.COLOR_RGB2HLS)
                sample_mask = sample.image[:,:,3] #alpha channel is a mask of the foreground
                hue_mask_raw = get_hue_mask(sample.image_hls[:,:,0], imaging_settings["hue_start"],imaging_settings["hue_end"])
                body_brightness_mask = cv.inRange(sample.image_hls[:,:,1],min(self.__hue_brightness_lower_limit,255-imaging_settings["number_brightness"]),max(self.__hue_brightness_upper_limit,255-imaging_settings["number_brightness"]))
                hue_mask = cv.bitwise_and(hue_mask_raw,body_brightness_mask)
                #hue_mask = cv.morphologyEx(hue_mask,cv.MORPH_OPEN,get_kernel(3),iterations=1)
                #hue_mask = cv.morphologyEx(hue_mask,cv.MORPH_CLOSE,get_kernel(3),iterations=1)
                die_body_mask = cv.morphologyEx(cv.bitwise_and(hue_mask,sample_mask),cv.MORPH_CLOSE,get_kernel(3),iterations=1)
                die_number_mask = cv.morphologyEx(cv.bitwise_and(cv.bitwise_not(hue_mask),cv.morphologyEx(sample_mask,cv.MORPH_ERODE,get_kernel(5),iterations=3)),cv.MORPH_OPEN,get_kernel(3),iterations=1) #remove noise at the edge of the sample area
                # die_body_mask = cv.morphologyEx(die_body_mask,cv.MORPH_ERODE,get_kernel(3),iterations=2)
                # die_number_mask = cv.morphologyEx(die_number_mask,cv.MORPH_ERODE,get_kernel(3),iterations=2)
                body_brightnesses = np.add(body_brightnesses,cv.calcHist([sample.image_hls[:,:,1]],[0], die_body_mask,[256],[0,256]))
                number_brightnesses = np.add(number_brightnesses,cv.calcHist([sample.image_hls[:,:,1]],[0], die_number_mask,[256],[0,256]))
                if log_level >= LOG_LEVEL_VERBOSE:
                    plt.subplot(2,4,1)
                    plt.imshow(sample.image)
                    plt.title("RGBA Image")
                    plt.subplot(2,4,2)
                    plt.imshow(sample.image_hls[:,:,0],cmap='gray')
                    plt.title("Image Hue")
                    plt.subplot(2,4,3)
                    plt.imshow(sample.image_hls[:,:,1],cmap='gray')
                    plt.title("Image Brightness")
                    plt.subplot(2,4,4)
                    plt.imshow(sample_mask,cmap='gray')
                    plt.title("Image Sample ROI")

                    plt.subplot(2,4,5)
                    plt.imshow(hue_mask_raw,cmap='gray')
                    plt.title("Mask for Hue Matching")
                    plt.subplot(2,4,6)
                    plt.imshow(hue_mask,cmap='gray')
                    plt.title("Filtered Mask for Hue Matching")
                    plt.subplot(2,4,7)
                    plt.imshow(die_body_mask,cmap='gray')
                    plt.title("Mask for Body ID")
                    plt.subplot(2,4,8)
                    plt.imshow(die_number_mask,cmap='gray')
                    plt.title("Mask for Number ID")
                    plt.show()
                

        body_brightnesses = body_brightnesses / np.sum(body_brightnesses)
        body_brightnesses_cum = np.cumsum(body_brightnesses)
        number_brightnesses = number_brightnesses / np.sum(number_brightnesses)
        number_brightnesses_cum = np.cumsum(number_brightnesses)

        body_brightnesses_cum_grad = np.gradient(body_brightnesses_cum)
        number_brightnesses_cum_grad = np.gradient(number_brightnesses_cum,)
        
        
        if number_brightnesses_cum[127] > 0.5: #text is black if at least 50% of the die is very dark
            imaging_settings["number_brightness"] = 0
            brightness_balance = np.abs(body_brightnesses_cum+number_brightnesses_cum-1)
            imaging_settings["brightness_threshold"] = int(np.argmin(brightness_balance))
        else:
            imaging_settings["number_brightness"] = 255
            brightness_balance = np.abs(1-body_brightnesses_cum-number_brightnesses_cum)
            balance_threshold = np.argmin(brightness_balance)
            number_threshold = np.argmax(number_brightnesses_cum_grad > self.__brightness_gradient_indicator)
            imaging_settings["brightness_threshold"] = int(max(balance_threshold,number_threshold))
        
        if log_level >= LOG_LEVEL_DEBUG:
            f = plt.figure()
            f.set_figheight(9)
            plt.subplot(3,1,1)
            plt.plot(body_brightnesses)
            plt.plot(number_brightnesses)
            plt.title("Brightness Histogram")
            plt.subplot(3,1,2)
            plt.plot(body_brightnesses_cum)
            plt.plot(number_brightnesses_cum)
            plt.title("Cumulative Brightness")
            plt.subplot(3,1,3)
            plt.plot(body_brightnesses_cum_grad)
            plt.plot(number_brightnesses_cum_grad)
            plt.title("Cumulative Brightness Gradient")
            plt.suptitle("Brightness Threshold = %u"%imaging_settings["brightness_threshold"])
            plt.show()

        return imaging_settings


        
    def get_color_mask_from_image(self, image_hls, log_level=LOG_LEVEL_FATAL):
        hue_mask = get_hue_mask(image_hls[:,:,0],self.imaging_settings["hue_start"],self.imaging_settings["hue_end"])
        # _, body_brightness_mask = get_brightness_masks(image_hls,self.imaging_settings["number_brightness"], self.imaging_settings["brightness_threshold"])
        _, brightness_mask = cv.threshold(image_hls[:,:,1],self.__backround_brightness_threshold,255, cv.THRESH_BINARY)
        body_mask = cv.bitwise_and(hue_mask, brightness_mask)
        # if self.imaging_settings["hue_start"] > self.imaging_settings["hue_end"]:
        #     mask = cv.inRange(image_hls,(0,self.__hue_brightness_threshold,self.imaging_settings["saturation_start"]),(self.imaging_settings["hue_end"],255,self.imaging_settings["saturation_end"]))
        #     mask = np.add(mask,cv.inRange(image_hls,(self.imaging_settings["hue_start"],self.__hue_brightness_threshold,self.imaging_settings["saturation_start"]),(180,255,self.imaging_settings["saturation_end"])))
        # else:
        #     mask = cv.inRange(image_hls,(self.imaging_settings["hue_start"],self.__hue_brightness_threshold,self.imaging_settings["saturation_start"]),(self.imaging_settings["hue_end"],255,self.imaging_settings["saturation_end"]))
        body_mask = cv.morphologyEx(body_mask,cv.MORPH_CLOSE,get_kernel(5))
        
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(body_mask, cmap='gray')
            plt.title("Mask Hue=[%u, %u], Brightness>%u"%(self.imaging_settings["hue_start"],self.imaging_settings["hue_end"],self.__backround_brightness_threshold))
            plt.show() 
        return draw_all_contours(body_mask,0.25,50,cv.FILLED,log_level)
    

    def get_best_point_from_masks(self, die_color_mask, die_location_mask, log_level=LOG_LEVEL_FATAL):
        die_mask =  cv.bitwise_and(die_color_mask, die_location_mask)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(die_mask,cmap='gray')
            plt.title("Combined Mask for Die Search")
            plt.show()
        return self.get_best_point_from_mask(die_mask, self.get_reference_radius(), log_level=log_level)

    @staticmethod
    def get_best_point_from_mask(mask, reference_radius, log_level=LOG_LEVEL_FATAL):
        dist_transform = cv.distanceTransform(mask,cv.DIST_L2,5)
        dist_transform_norm = np.abs(dist_transform.astype(np.float32) - reference_radius) / reference_radius
        minVal, maxVal, minP, maxP = cv.minMaxLoc(dist_transform_norm)
        confidence = 1-minVal
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(dist_transform,cmap='gray')
            plt.title("Distance Transform")
            plt.show()
        if log_level >= LOG_LEVEL_DEBUG:
            plt.imshow(dist_transform_norm,cmap='gray')
            plt.title("Normalized Distance Transform (lowest error %u%%)"%int(minVal*100))
            plt.show()
        return minP, confidence
    
    def get_reference_radius(self):
        return self.imaging_settings["average_circumscribed_radius"]*self.geometry["enscribed_perimeter_ratio"]

    #confidence should be 1 when max dist_transform is the same as enscribed periemter
    def get_best_point_from_image(self, image_RGB, mask=None, image_HLS=None, log_level=LOG_LEVEL_FATAL):
        if image_HLS is None:
            image_HLS = cv.cvtColor(image_RGB,cv.COLOR_RGB2HLS)
        color_mask = self.get_color_mask_from_image(image_HLS,log_level)
        if mask is not None:
            if log_level >= LOG_LEVEL_VERBOSE:
                plt.imshow(mask,cmap='gray')
                plt.title("Provided Mask for Die Search")
                plt.show()
            return self.get_best_point_from_masks(color_mask, mask, log_level=log_level)
        else:
            reference_radius = self.get_reference_radius()
            return self.get_best_point_from_mask(color_mask,reference_radius,log_level)
    
    def get_sample_at_point_from_image(self, image_RGB, point, image_HLS=None, mask=None, log_level=LOG_LEVEL_FATAL):
        if image_HLS is None:
            image_HLS = cv.cvtColor(image_RGB,cv.COLOR_RGB2HLS)
        if mask is None:
            mask = self.get_color_mask_from_image(image_HLS,log_level)
        return DieFaceSample(DieFaceSample.crop_sample_from_image(image_RGB, mask, point, self.imaging_settings["average_circumscribed_radius"],log_level),self.geometry, self.imaging_settings,log_level)
    
    
        
    def get_best_face_match_from_sample(self, sample, camera_matrix=None, scoring_function=get_scores_from_results, log_level=LOG_LEVEL_FATAL):
        results = self.compare_to(sample,camera_matrix,log_level)
        scores = scoring_function(results,log_level)
        for i in np.arange(0,len(results)):
            results[i]["score"] = scores[i]
        best_index = np.argmax(scores)
        return results[best_index]["face_value"], scores[best_index], results, scores, best_index
    
    # check that all samples of all OTHER faces do not match any samples from each face
    def compare_faces(self, camera_matrix, log_level=LOG_LEVEL_FATAL):
        scores = list()
        results = list()
        keys = list(self.faces.keys())
        for i in np.arange(0,len(self.faces)-1):
            for si in np.arange(0,len(self.faces[keys[i]].samples)):
                for j in np.arange(i+1,len(self.faces)):
                    for sj in np.arange(0,len(self.faces[keys[j]].samples)):
                        result = self.faces[keys[i]].samples[si].compare_to(self.faces[keys[j]].samples[sj],camera_matrix,log_level)
                        result_scores = get_scores_from_results([result],log_level)
                        scores.append(result_scores[0])
                        results.append((self.faces[keys[i]].samples[si],self.faces[keys[j]].samples[sj],result))
                        if result_scores[0] > 0.25 and log_level >= LOG_LEVEL_WARN:
                            print("WARNING: samples from different faces (face %u sample %u vs face %u sample %u) have a high similarity (%u)"%(self.faces[keys[i]].value,si,self.faces[keys[j]].value,sj,result_scores[0]))

        return scores, results

    def compare_hu_moments(self, log_level=LOG_LEVEL_FATAL):
        hu_moments = list()
        face_names = list(self.faces.keys())
        for h in np.arange(0,7):
            hu_moment_values = list()
            for face_name in face_names:
                hu_moment_values.append(np.array([s.moments[0][2][h] for s in self.faces[face_name].samples]))
            if log_level >= LOG_LEVEL_DEBUG:
                plt.subplot(2,4,h+1)
                plt.boxplot(hu_moment_values,0,'')
                #plt.title("Hu Moment #%u"%(h+1))
            hu_moments.append(hu_moment_values)
        if log_level >= LOG_LEVEL_DEBUG:
            plt.show()
        return hu_moments
    
    def compare_zernike_moments(self, log_level=LOG_LEVEL_FATAL):
        z_moments = list()
        face_names = list(self.faces.keys())
        for z in np.arange(0,24):
            z_moment_values = list()
            for face_name in face_names:
                z_moment_values.append(np.array([s.moments[0][3][z] for s in self.faces[face_name].samples]))
            if log_level >= LOG_LEVEL_DEBUG:
                plt.subplot(4,6,z+1)
                plt.boxplot(z_moment_values,0,'')
                #plt.title("Zernike Moment #%u"%(z+1))
            z_moments.append(z_moment_values)
        if log_level >= LOG_LEVEL_DEBUG:
            plt.show()
        return z_moments




    @staticmethod
    def get_common_die_geometry(rank):
        g = dict.fromkeys(Die.geometry_keys)
        match rank:
            case 1:
                g["enscribed_perimeter_ratio"] = 1.0
                g["circumscribed_face_ratio"] = 1.0
                g["enscribed_face_ratio"] = 1.0
                g["pixel_comparison_ratio"] = g["enscribed_face_ratio"]
                g["perimeter_edges"] = 1
                g["face_edges"] = 1
                g["adjacent_faces"] = 0
            case 4:
                # tetrahedron:
                # g["enscribed_perimeter_ratio"] = math.cos(math.pi/3)
                # g["circumscribed_face_ratio"] = 1
                # g["enscribed_face_ratio"] = math.cos(math.pi/3)
                # g["pixel_comparison_ratio"] = g["enscribed_face_ratio"]
                # g["perimeter_edges"] = 3
                # g["face_edges"] = 3

                # tombstone:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/4)
                g["circumscribed_face_ratio"] = 1.0
                g["enscribed_face_ratio"] = math.cos(math.pi/4)
                g["pixel_comparison_ratio"] = g["enscribed_face_ratio"]
                g["perimeter_edges"] = 4
                g["face_edges"] = 4
                g["adjacent_faces"] = 0
            case 6:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/4)
                g["circumscribed_face_ratio"] = 1.0
                g["enscribed_face_ratio"] = math.cos(math.pi/4)
                g["pixel_comparison_ratio"] = g["enscribed_face_ratio"]
                g["perimeter_edges"] = 4
                g["face_edges"] = 4
                g["adjacent_faces"] = 0
            case 8:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/6)
                g["circumscribed_face_ratio"] = 1.0
                g["enscribed_face_ratio"] = math.cos(math.pi/3)
                g["pixel_comparison_ratio"] = g["enscribed_face_ratio"]
                g["perimeter_edges"] = 6
                g["face_edges"] = 3
                g["adjacent_faces"] = 0
            case 10:
                g["enscribed_perimeter_ratio"] = 0.747
                g["circumscribed_face_ratio"] = 0.725
                g["enscribed_face_ratio"] = 0.5# 0.365 #force bigger to account for offset
                g["pixel_comparison_ratio"] = 0.95 #want to include both edge numbers
                g["perimeter_edges"] = 6
                g["face_edges"] = 4
                g["adjacent_faces"] = 2
            case 12:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/10)
                g["circumscribed_face_ratio"] = 0.618
                g["enscribed_face_ratio"] = 0.500
                g["pixel_comparison_ratio"] = g["enscribed_face_ratio"]
                g["perimeter_edges"] = 10
                g["face_edges"] = 5
                g["adjacent_faces"] = 0
            case 20:
                g["enscribed_perimeter_ratio"] = math.cos(math.pi/6)
                g["circumscribed_face_ratio"] = 0.619
                g["enscribed_face_ratio"] = 0.313
                g["pixel_comparison_ratio"] = 0.75 #try to include 3 close faces but reject other 6 high angle faces
                g["perimeter_edges"] = 6
                g["face_edges"] = 3
                g["adjacent_faces"] = 3
            case _:
                raise ValueError("Unsupported Die Rank for Default Geometry")
        return g

    @staticmethod
    def get_common_die_face_values(rank, start=1, step=1):
        return np.arange(start,(rank*step)+start,step)

    @staticmethod
    def get_common_die_info(name):
        match name:
            case 4:
                return (4,1,1)
            case 6:
                return (6,1,1)
            case 8:
                return (8,1,1)
            case 10:
                return (10,0,1)
            case 12:
                return (12,1,1)
            case 20:
                return (20,1,1)
            case 100:
                return (10,0,10)
            case _:
                raise ValueError("Unsupported Die Name for Default Layout")
