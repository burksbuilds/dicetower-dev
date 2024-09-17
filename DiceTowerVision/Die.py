import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision import DieFace
from DiceTowerVision import DieFaceSample
from DiceTowerVision import DieFaceTemplate
from DiceTowerVision import DieImagingSettings
from DiceTowerVision import DieGeometry
from DiceTowerVision import ContourGroupMatcher
from DiceTowerVision import MatchResultSettings
from collections import namedtuple
import math
from dataclasses import dataclass, field

@dataclass
class DieSearchResult:
    die:object = field(default=None)
    point:object = field(default_factory=np.zeros((1,2)))
    confidence:float = field(default=0)

class Die:
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
    __hue_detection_margin = 5

    def __init__(self, name, face_values, face_templates, imaging_settings, geometry=None, log_level=LOG_LEVEL_FATAL):
        self.name = str(name)
        self.faces = dict()
        for i in np.arange(0,len(face_values)):
            self.faces[str(face_values[i])] = DieFace(self, face_values[i], face_templates[i])
            if face_values[i] == 0: #rotationally symmetric face
                self.faces[str(face_values[i])].template.match_result_settings.is_rotationally_symmetric = True
        self.geometry = geometry
        if self.geometry is None:
            self.geometry = Die.get_common_die_geometry(len(face_values))
        self.imaging_settings = imaging_settings

    @staticmethod
    def create_from_samples(name, samples, face_values, geometry=None, imaging_settings=None, match_settings=None, log_level=LOG_LEVEL_FATAL):
        if geometry is None:
            geometry = Die.get_common_die_geometry(len(face_values))
        if imaging_settings is None:
            imaging_settings = DieImagingSettings.get_imaging_settings(samples, log_level=log_level)
        templates = [DieFaceTemplate.create_from_sample(s,imaging_settings,match_settings,log_level=log_level) for s in samples]
        return Die(name, face_values, templates, imaging_settings, geometry, log_level=log_level)


    @staticmethod
    def create_from_images(file_template, name, face_values, geometry, imaging_settings=None, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        samples = list()
        for fv in np.arange(1,len(face_values)+1):
            image_path = file_template.replace("FV#",str(fv))
            samples.append(DieFaceSample.create_from_image(image_path,geometry,autocrop, roi_mask=roi_mask, log_level=log_level))
        return Die.create_from_samples(name, samples, face_values, geometry, imaging_settings, log_level=log_level)

    
    @staticmethod
    def create_common_die_from_images(file_template, name, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        face_values = Die.get_common_die_face_values(name)
        return Die.create_from_images(file_template,name, face_values,DieGeometry.get_common_die_geometry(len(face_values)) ,roi_mask=roi_mask, log_level=log_level)
    

        
    def get_color_mask_from_image(self, image_hls, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        hues = image_hls[:,:,0]
        if roi_mask is not None:
            hues = cv.bitwise_and(hues, roi_mask)
        hue_mask = get_hue_mask(hues,self.imaging_settings.hue_start,self.imaging_settings.hue_end, self.__hue_detection_margin)
        # _, body_brightness_mask = get_brightness_masks(image_hls,self.imaging_settings["number_brightness"], self.imaging_settings["brightness_threshold"])
        _, brightness_mask = cv.threshold(image_hls[:,:,1],DieFaceSample.backround_brightness_threshold,255, cv.THRESH_BINARY)
        body_mask = cv.bitwise_and(hue_mask, brightness_mask)
        # if self.imaging_settings["hue_start"] > self.imaging_settings["hue_end"]:
        #     mask = cv.inRange(image_hls,(0,self.__hue_brightness_threshold,self.imaging_settings["saturation_start"]),(self.imaging_settings["hue_end"],255,self.imaging_settings["saturation_end"]))
        #     mask = np.add(mask,cv.inRange(image_hls,(self.imaging_settings["hue_start"],self.__hue_brightness_threshold,self.imaging_settings["saturation_start"]),(180,255,self.imaging_settings["saturation_end"])))
        # else:
        #     mask = cv.inRange(image_hls,(self.imaging_settings["hue_start"],self.__hue_brightness_threshold,self.imaging_settings["saturation_start"]),(self.imaging_settings["hue_end"],255,self.imaging_settings["saturation_end"]))
        body_mask = cv.morphologyEx(body_mask,cv.MORPH_CLOSE,get_kernel(5))
        
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(body_mask, cmap='gray')
            plt.title("Mask Hue=[%u, %u], Brightness>=%u"%(self.imaging_settings.hue_start-self.__hue_detection_margin,self.imaging_settings.hue_end+self.__hue_detection_margin,self.__backround_brightness_threshold))
            plt.show() 
        return draw_all_contours(body_mask,0.25,50,cv.FILLED,log_level)

    def get_best_point_from_mask(self,mask, reference_radius=None, log_level=LOG_LEVEL_FATAL) -> DieSearchResult:
        dist_transform = cv.distanceTransform(mask,cv.DIST_L2,5)
        if reference_radius is None:
            reference_radius = self.get_reference_radius()
        dist_transform_norm = np.abs(dist_transform.astype(np.float32) - reference_radius) / reference_radius
        minVal, _, minP, _ = cv.minMaxLoc(dist_transform_norm)
        result = DieSearchResult(die=self,point=minP,confidence=1-minVal)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.subplot(1,2,1)
            plt.imshow(mask,cmap='gray')
            plt.title("Mask for Die Search")
            plt.subplot(1,2,2)
            plt.imshow(dist_transform,cmap='gray')
            plt.title("Distance Transform")
            plt.show()
        if log_level >= LOG_LEVEL_DEBUG:
            plt.imshow(dist_transform_norm,cmap='gray')
            plt.title("Normalized Distance Transform (lowest error %u%%)"%int(minVal*100))
            plt.show()
        return result
    
    def get_reference_radius(self):
        return self.imaging_settings.average_circumscribed_radius*self.geometry.enscribed_perimeter_ratio

    #confidence should be 1 when max dist_transform is the same as enscribed periemter
    def get_best_point_from_image(self, image_RGB, mask=None, image_HLS=None, log_level=LOG_LEVEL_FATAL) -> DieSearchResult:
        if image_HLS is None:
            image_HLS = cv.cvtColor(image_RGB,cv.COLOR_RGB2HLS)
        color_mask = self.get_color_mask_from_image(image_HLS,mask,log_level)
        reference_radius = self.get_reference_radius()
        return self.get_best_point_from_mask(color_mask,reference_radius,log_level)
    
    def get_sample_at_point_from_image(self, image_RGB, point, image_HLS=None, mask=None, log_level=LOG_LEVEL_FATAL):
        if image_HLS is None:
             image_HLS = cv.cvtColor(image_RGB,cv.COLOR_RGB2HLS)
        # if mask is None:
        #     mask = self.get_color_mask_from_image(image_HLS,log_level=log_level)
        return DieFaceSample(DieFaceSample.crop_sample_from_image(image_RGB, mask, point, self.imaging_settings.average_circumscribed_radius,log_level),self.geometry,log_level)
    

    
    def compare_to_image(self, other, keypoints=None, descriptors=None, log_level=LOG_LEVEL_FATAL):
        if keypoints is None or descriptors is None:
            keypoints, descriptors = ContourGroupMatcher.get_keypoints_and_descriptors(other, log_level=log_level)
        return [face.compare_to_image(other,keypoints,descriptors,log_level=log_level) for face in self.faces.values()]
        
    
    def compare_to_sample(self, sample, log_level=LOG_LEVEL_FATAL):
        sample_image = sample.get_keypoint_detection_image(self.imaging_settings, log_level=log_level)
        sample_keypoints, sample_descriptors = ContourGroupMatcher.get_keypoints_and_descriptors(sample_image, log_level=log_level)
        return self.compare_to_image(sample_image, sample_keypoints, sample_descriptors, log_level=log_level)


    def compare_faces(self, log_level=LOG_LEVEL_FATAL):
        scores = np.zeros((len(self.faces)-1,len(self.faces)))
        results = list()

        for i, this_face in enumerate(self.faces.values()):
            for j, other_face in enumerate(self.faces.values()):
                result = this_face.compare_to_image(other_face.template.image, log_level=log_level)
                results.append(result)
                if other_face != this_face:
                    scores[i,j] = result.confidence
                    if log_level == LOG_LEVEL_INFO and result.confidence > 0.5:
                        result.view(other_face.template.image)
            if log_level >= LOG_LEVEL_INFO:
                self.view_match_results(results[i*(len(self.faces)):(i+1)*(len(self.faces))],"Face #%u"%this_face.value)
        
        if log_level >= LOG_LEVEL_INFO:
            face_names = list(self.faces.keys())
            plt.boxplot(scores,tick_labels=face_names)
            plt.ylim((0,1))
            plt.show()
            
        return results

    def view_templates(self):
        rows = int(np.sqrt(len(self.faces)))
        cols = int(1+len(self.faces)/rows)

        i = 1
        for face in self.faces.values():
            plt.subplot(rows,cols,i)
            plt.axis('off')
            image_RGB = cv.cvtColor(face.template.top_face_contours.image,cv.COLOR_GRAY2RGB)
            channel = 0
            for a in face.template.adjacent_face_contours:
                image_RGB[:,:,channel] = cv.bitwise_or(image_RGB[:,:,channel],a.image)
                channel = np.mod(channel+1,3)
            plt.imshow(image_RGB)
            plt.title(str(face.value))
            i += 1
        plt.gcf().set_size_inches((2*cols,2*rows))
        plt.suptitle("Faces of %s"%self.name)
        plt.show()

    def view_match_results(self, match_results, context=""):
        labels = [str(r.face.value) for r in match_results]
        rows = 3+self.geometry.adjacent_faces
        x = np.arange(0,len(labels))

        plt.subplots(rows,2,layout="constrained")

        plt.subplot(rows,2,1)
        plt.title("Confidence")
        bar_ax = plt.bar(x,np.array([r.confidence for r in match_results]),tick_label=labels)
        plt.ylim(0,1)
        plt.bar_label(bar_ax, fmt="%2.2f")

        plt.subplot(rows,2,2)
        plt.title("Confidences")
        confidence_labels = ["Top Face", "Top Face (Eroded)"]
        for i in np.arange(0,len(match_results[0].confidences)-2,2):
            confidence_labels.append("Adj Face #%u"%(i+1))
            confidence_labels.append("Adj Face #%u (Eroded)"%(i+1))
        bottom = np.zeros(len(labels))
        for i, label in enumerate(confidence_labels):
            vals = np.array([r.confidences[i] for r in match_results])
            plt.bar(labels, vals, label=label, bottom=bottom)
            bottom += vals
        plt.ylim(0,1)

        plt.subplot(rows,2,3)
        plt.title("Top Face")
        scores = np.array([r.top_face_scores[0] for r in match_results])
        plt.bar(x,scores,tick_label=labels)
        if np.any(np.isinf(scores)):
            plt.scatter(x[np.isinf(scores)],np.zeros(np.count_nonzero(np.isinf(scores))),c='r',marker='X')
        plt.ylim(0,10)
        
        plt.subplot(rows,2,4)
        plt.title("Top Face (Eroded)")
        scores = np.array([r.top_face_scores[1] for r in match_results])
        plt.bar(x,scores,tick_label=labels)
        if np.any(np.isinf(scores)):
            plt.scatter(x[np.isinf(scores)],np.zeros(np.count_nonzero(np.isinf(scores))),c='r',marker='X')
        plt.bar(x,scores,tick_label=labels)
        plt.ylim(0,10)

        plt.subplot(rows,2,5)
        plt.title("Top Face Reverse")
        scores = np.array([r.top_face_matched_scores[0] for r in match_results])
        plt.bar(x,scores,tick_label=labels)
        if np.any(np.isinf(scores)):
            plt.scatter(x[np.isinf(scores)],np.zeros(np.count_nonzero(np.isinf(scores))),c='r',marker='X')
        plt.ylim(0,10)
        
        plt.subplot(rows,2,6)
        plt.title("Top Face Reverse (Eroded)")
        scores = np.array([r.top_face_matched_scores[1] for r in match_results])
        plt.bar(x,scores,tick_label=labels)
        if np.any(np.isinf(scores)):
            plt.scatter(x[np.isinf(scores)],np.zeros(np.count_nonzero(np.isinf(scores))),c='r',marker='X')
        plt.bar(x,scores,tick_label=labels)
        plt.ylim(0,10)

        for a in np.arange(0,self.geometry.adjacent_faces):  
            plt.subplot(rows,2,7+2*a)
            plt.title("Adjacent Face #%u"%a)
            scores = np.array([r.adj_face_scores[a][0] for r in match_results])
            plt.bar(x,scores,tick_label=labels)
            if np.any(np.isinf(scores)):
                plt.scatter(x[np.isinf(scores)],np.zeros(np.count_nonzero(np.isinf(scores))),c='r',marker='X')
            plt.bar(x,scores,tick_label=labels)
            plt.ylim(0,10)

            plt.subplot(rows,2,8+2*a)
            plt.title("Adjacent Face #%u (Eroded)"%a)
            scores = np.array([r.adj_face_scores[a][1] for r in match_results])
            plt.bar(x,scores,tick_label=labels)
            if np.any(np.isinf(scores)):
                plt.scatter(x[np.isinf(scores)],np.zeros(np.count_nonzero(np.isinf(scores))),c='r',marker='X')
            plt.bar(x,scores,tick_label=labels)
            plt.ylim(0,10)

        plt.suptitle("Die Match Results %s"%context)
        plt.gcf().set_size_inches((6+0.1875*len(labels),4+1.5*self.geometry.adjacent_faces))
        plt.show()




    @staticmethod
    def get_common_die_face_values(name):
        match name:
            case 4:
                return Die.get_equidistant_face_values(4,1,1)
            case 6:
                return Die.get_equidistant_face_values(6,1,1)
            case 8:
                return Die.get_equidistant_face_values(8,1,1)
            case 10:
                return Die.get_equidistant_face_values(10,0,1)
            case 12:
                return Die.get_equidistant_face_values(12,1,1)
            case 20:
                return Die.get_equidistant_face_values(20,1,1)
            case 100:
                return Die.get_equidistant_face_values(10,0,10)
            case _:
                raise ValueError("Unsupported Die Name for Default Layout")

    @staticmethod
    def get_equidistant_face_values(rank, start=1, step=1):
        return np.arange(start,(rank*step)+start,step)

    # @staticmethod
    # def get_common_die_info(name):
    #     match name:
    #         case 4:
    #             return (4,1,1)
    #         case 6:
    #             return (6,1,1)
    #         case 8:
    #             return (8,1,1)
    #         case 10:
    #             return (10,0,1)
    #         case 12:
    #             return (12,1,1)
    #         case 20:
    #             return (20,1,1)
    #         case 100:
    #             return (10,0,10)
    #         case _:
    #             raise ValueError("Unsupported Die Name for Default Layout")
