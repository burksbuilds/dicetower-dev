import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision.ContourGroup import *
from DiceTowerVision.ContourGroupMatcher import *
from dataclasses import dataclass, field

@dataclass
class DieFaceMatchResult:
    face:object = field(default=None, )
    confidence:float = field(default=0.0)
    confidences:np.ndarray = field(default_factory=lambda:np.zeros(2))
    top_face_match:ContourGroupSearchResult = field(default=None)
    top_face_scores:tuple[float,float] = field(default=(np.inf, np.inf))
    top_face_matched_contours:ContourGroup = field(default=None)
    top_face_matched_scores:tuple[float,float] = field(default=(np.inf, np.inf))
    adj_face_matches:list[ContourGroupSearchResult] = field(default_factory= lambda: list())
    adj_face_scores:list[tuple[float,float]] = field(default_factory = lambda: list())

    def transform(self, image):
        if self.top_face_match.affine_warp is None or self.face is None:
            return image
        image_warp_full = cv.warpAffine(image, self.top_face_match.affine_warp, self.face.template.image.shape)
        image_warp_composite = cv.bitwise_and(image_warp_full,self.face.template.top_face_mask)
        for i, m in enumerate(self.adj_face_matches):
            if m.affine_warp is not None:
                m_warp = cv.warpAffine(image_warp_full, m.affine_warp, self.face.template.image.shape)
            else:
                m_warp = image_warp_full
            image_warp_composite = cv.bitwise_or(image_warp_composite,cv.bitwise_and(self.face.template.adjacent_face_masks[i], m_warp))
        return image_warp_composite
    
    def view(self, image):
        image_warp = self.transform(image)
        image_diff = np.zeros((image_warp.shape[0], image_warp.shape[1],3), dtype=np.uint8)
        image_diff[:,:,0] = self.face.template.image
        image_diff[:,:,1] = image_warp
        image_diff[:,:,2] = cv.bitwise_and(self.face.template.image, image_warp)

        #plt.subplots(3,3,layout="constrained")
        rows = 3 + int((len(self.adj_face_matches)+1)/2)

        plt.subplot(rows,4,4)
        plt.imshow(self.face.template.image, cmap='gray')
        plt.title("Template")
        plt.subplot(rows,4,3)
        plt.imshow(image_diff)
        plt.title("Comparison")
        plt.subplot(rows,4,2)
        plt.imshow(image_warp, cmap='gray')
        plt.title("Sample (Warp)")
        plt.subplot(rows,4,1)
        plt.imshow(image, cmap='gray')
        plt.title("Sample (Orig)")

        plt.subplot(rows,4,7)
        plt.imshow(cv.bitwise_xor(self.face.template.image, image_warp), cmap='gray')
        plt.title("Difference")
        plt.subplot(rows,4,8)
        plt.imshow(cv.morphologyEx(cv.bitwise_xor(self.face.template.image, image_warp), cv.MORPH_ERODE, get_kernel(3)), cmap='gray')
        plt.title("Difference (E)")

        plt.subplot2grid((rows,4),(1,0),colspan=2)
        image_match = cv.drawMatches(image,self.top_face_match.matches_used_keypoints,self.face.template.top_face_matcher.group.image,self.face.template.top_face_matcher.keypoints,self.top_face_match.matches_used_list,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(image_match)
        plt.title("Keypoints")

        plt.subplot2grid((rows,4),(2,0),colspan=4)
        labels = ["Top", "Top (B)", "Top (E)", "Top (BE)"]
        scores = np.array([self.top_face_scores[0], self.top_face_matched_scores[0], self.top_face_scores[1], self.top_face_matched_scores[1]])
        for i, a in enumerate(self.adj_face_scores):
            labels.append("#%u"%(i))
            labels.append("#%u (E)"%i)
            scores = np.append(scores, [a[0],a[1]])
        x = np.arange(0,len(labels))
        bar_ax = plt.bar(x, scores, tick_label=labels)
        if np.any(np.isinf(scores)):
            plt.scatter(x[np.isinf(scores)],np.zeros(np.count_nonzero(np.isinf(scores))),c='r',marker='X')
        plt.ylim((0,10))
        plt.bar_label(bar_ax, fmt="%2.2f")

        if self.top_face_match.affine_warp is None:
            image_warp_partial = image
        else:
            image_warp_partial = cv.warpAffine(image,self.top_face_match.affine_warp,self.face.template.image.shape)
        for i in np.arange(0,len(self.adj_face_matches)):
            row = 3 + int(i/2)
            col = 2* int(np.mod(i,2))
            plt.subplot2grid((rows,4),(row,col),colspan=2)
            image_match = cv.drawMatches(image_warp_partial,self.adj_face_matches[i].matches_used_keypoints,self.face.template.adjacent_face_matchers[i].group.image,self.face.template.adjacent_face_matchers[i].keypoints,self.adj_face_matches[i].matches_used_list,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(image_match)
            plt.title("Keypoints #%u"%i)

        plt.gcf().set_size_inches(10,2*rows)
        plt.suptitle("Match Result (Confidence = %2.0f%%)"%(self.confidence*100))
        plt.show()

    def to_flat_dict(self, rank):
        d = {
            #'die' : '' if self.face is None else self.face.die.name,
            'face' : '' if self.face is None else str(self.face.value),
            'face_confidence' : self.confidence,
            'face_rank' : rank,
            'top_score' : '' if np.isinf(self.top_face_scores[0]) else self.top_face_scores[0],
            'top_score_eroded' : '' if np.isinf(self.top_face_scores[1]) else self.top_face_scores[1],
            # 'top_confidence' : self.confidences[0],
            # 'top_confidence_eroded' : self.confidences[1],
            'top_match_angle' : '' if self.top_face_match.affine_warp is None else self.top_face_match.affine_warp_components.angle,
            'top_match_shear' : '' if self.top_face_match.affine_warp is None else self.top_face_match.affine_warp_components.shear,
            'top_match_scale_x' : '' if self.top_face_match.affine_warp is None else self.top_face_match.affine_warp_components.scale_x,
            'top_match_scale_y' : '' if self.top_face_match.affine_warp is None else self.top_face_match.affine_warp_components.scale_y

        }
        for i in np.arange(0,len(self.adj_face_scores)):
            d["adjacent_%u_score"%(i+1)] = '' if np.isinf(self.adj_face_scores[i][0]) else self.adj_face_scores[i][0]
            d["adjacent_%u_score_eroded"%(i+1)] = '' if np.isinf(self.adj_face_scores[i][1]) else self.adj_face_scores[i][1]
            # d["adjacent_%u_confidence"%(i+1)] = self.confidences[2*i+2]
            # d["adjacent_%u_confidence_eroded"%(i+1)] = self.confidences[2*i+3]
            d["adjacent_%u_match_angle"%(i+1)] = '' if self.adj_face_matches[i].affine_warp is None else self.adj_face_matches[i].affine_warp_components.angle
            d["adjacent_%u_match_shear"%(i+1)] = '' if self.adj_face_matches[i].affine_warp is None else self.adj_face_matches[i].affine_warp_components.shear
            d["adjacent_%u_match_scale_x"%(i+1)] = '' if self.adj_face_matches[i].affine_warp is None else self.adj_face_matches[i].affine_warp_components.scale_x
            d["adjacent_%u_match_scale_y"%(i+1)] = '' if self.adj_face_matches[i].affine_warp is None else self.adj_face_matches[i].affine_warp_components.scale_y
        return d
    
    @staticmethod
    def get_flat_dict_field_names(num_adjacent_faces):
        names = ['face', 'face_confidence', 'face_rank']
        for i in np.arange(0,num_adjacent_faces+1):
            prefix = 'top_' if i == 0 else "adjacent_%u_"%i
            names.append(prefix+"score")
            names.append(prefix+"score_eroded")
            # names.append(prefix+"confidence")
            # names.append(prefix+"confidence_eroded")
            names.append(prefix+"match_angle")
            names.append(prefix+"match_shear")
            names.append(prefix+"match_scale_x")
            names.append(prefix+"match_scale_y")
        return names


@dataclass
class MatchResultSettings:
    top_face_offset:float = field(default=1.5)
    top_face_eroded_offset:float = field(default=0.5)
    top_face_scale:float = field(default=0.15)
    top_face_weight:float = field(default=1) 
    top_face_eroded_weight:float = field(default=1)
    adj_face_offset:float = field(default=1.5)
    adj_face_eroded_offset:float = field(default=0.5)
    adj_face_scale:float = field(default=0.15)
    adj_face_weight:float = field(default=1.0) #total for all adjacent faces
    adj_face_eroded_weight:float = field(default=1.0)
    is_rotationally_symmetric:bool = field(default=False)
    


class DieFaceTemplate:

    face_mask_min_margin = 15
    face_mask_max_margin = 50 #represents spacing between different contours of the same face 9two number or number and orientation symbol)
    scale_error_threshold = 0.25

    def __init__(self, image:np.ndarray, geometry, circumscribed_radius:float, match_settings=None, log_level=LOG_LEVEL_FATAL):
        self.image = image #BW image of relevant contours
        self.geometry = geometry
        self.circumscribed_radius = circumscribed_radius
        self.center = (self.image.shape[1]//2, self.image.shape[0]//2)
        if match_settings is None:
            self.match_result_settings = MatchResultSettings()
        else:
            self.match_result_settings = match_settings

        self.combined_contours = ContourGroup(self.image, log_level=log_level)
        self.individual_contours = self.combined_contours.split(log_level=log_level)

        #top_face_radius = self.circumscribed_radius * geometry.top_face_ratio
        self.top_face_contours = ContourGroup.combine([g for g in self.individual_contours if g.centroid_radius <= self.circumscribed_radius * geometry.top_face_ratio])
        other_contours = [g for g in self.individual_contours if g.centroid_radius > self.circumscribed_radius * geometry.top_face_ratio and 
                                                                          g.centroid_radius <= self.circumscribed_radius * geometry.adjacent_face_ratio]
        # some features (dots and underscores) of the top face might be technically outside the top_face_ratio, but they should be closer to the centroid of the top face feature than an enscribed face radius
        # this mostly helps out with D10s where the number is offset 'down' by a bit
        self.top_face_contours, other_contours = ContourGroup.group_by_distance(self.top_face_contours, other_contours, self.circumscribed_radius * geometry.top_face_ratio)
        self.top_face_center, self.top_face_radius = cv.minEnclosingCircle(self.top_face_contours.all_points())
        self.top_face_center = (int(self.top_face_center[0]),int(self.top_face_center[1]))

        self.top_face_matcher = ContourGroupMatcher(self.top_face_contours,log_level=log_level)
        self.top_face_mask = self.get_polar_mask(self.top_face_center,0,self.circumscribed_radius * geometry.enscribed_face_ratio,0,360,log_level=log_level)
        #cv.bitwise_and(self.top_face_contours.get_elliptical_mask(self.face_mask_max_margin,log_level=log_level),
                              #              cv.bitwise_or(self.top_face_contours.get_elliptical_mask(self.face_mask_min_margin,log_level=log_level),
                              #                          self.get_polar_mask(0,self.circumscribed_radius * geometry.top_face_ratio,0,360,log_level=log_level)))

        if geometry.adjacent_faces > 0:
            self.adjacent_face_contours = ContourGroup.group_by_angle(other_contours,geometry.adjacent_faces, self.circumscribed_radius * geometry.enscribed_face_ratio,log_level=log_level)
            if len(self.adjacent_face_contours) != geometry.adjacent_faces:
                if log_level >= LOG_LEVEL_WARN:
                    print("WARNING: Template does not have enough adjacent face contours!")
            self.adjacent_face_matchers = [ContourGroupMatcher(cg, log_level=log_level) for cg in self.adjacent_face_contours]
            adjacent_face_mask_halfwidth = min(45,(360 / geometry.adjacent_faces) / 2)
            self.adjacent_face_masks = [cv.bitwise_and(cg.get_elliptical_mask(self.face_mask_max_margin, log_level=log_level),
                                                       cv.bitwise_or(cg.get_elliptical_mask(self.face_mask_min_margin, log_level=log_level),
                                                                    self.get_polar_mask(self.top_face_center,
                                                                                        self.circumscribed_radius * geometry.top_face_ratio,
                                                                                        self.circumscribed_radius * geometry.adjacent_face_ratio, 
                                                                                        int(np.mod(cg.centroid_angle-adjacent_face_mask_halfwidth,360)), 
                                                                                        int(np.mod(cg.centroid_angle+adjacent_face_mask_halfwidth,360)), log_level=log_level
                                                                                        ))) for cg in self.adjacent_face_contours]
        else:
            self.adjacent_face_contours = list()
            self.adjacent_face_matchers = list()
            self.adjacent_face_masks = list()
        
        if log_level >= LOG_LEVEL_VERBOSE:
            self.view()

    def compare_to_image(self, image, keypoints=None, descriptors=None, log_level=LOG_LEVEL_FATAL):
        result = DieFaceMatchResult()
        result.adj_face_scores = [(np.inf,np.inf) for _ in np.arange(0,self.geometry.adjacent_faces)]
        result.adj_face_matches = [ContourGroupSearchResult() for _ in np.arange(0,self.geometry.adjacent_faces)]
        result.confidences = self.get_match_result_confidences(result) #just initialize

        # if keypoints is None or descriptors is None:
        #     keypoints, descriptors = ContourGroupMatcher.get_keypoints_and_descriptors(image, log_level=log_level)
        
        result.top_face_match = self.top_face_matcher.find_in_image(image, keypoints=keypoints, descriptors=descriptors, allow_distortion=False, log_level=log_level)
        if result.top_face_match.affine_warp is None and result.top_face_match.matches_used == 0:
            return result
        if abs(result.top_face_match.affine_warp_components.scale_x - 1) > 0.25:
            if log_level >= LOG_LEVEL_VERBOSE:
                print("WARNING: top face match found with out of bounds scale: %f"%result.top_face_match.affine_warp_components.scale_x)
            return result
        
        image_warped = cv.warpAffine(image,result.top_face_match.affine_warp, (self.image.shape[1], self.image.shape[0]))
        result.top_face_scores = self.top_face_matcher.compare_contour_area(image_warped,self.top_face_mask,log_level=log_level)

        #only bother with detailed scoring if the match has a decent score
        if result.top_face_scores[0] < 10 or log_level >= LOG_LEVEL_DEBUG:
            result.top_face_matched_contours = ContourGroup(cv.bitwise_and(image_warped,self.top_face_mask), log_level=log_level)
            top_face_matched_matcher = ContourGroupMatcher(result.top_face_matched_contours, log_level=log_level)
            result.top_face_matched_scores = top_face_matched_matcher.compare_contour_area(self.top_face_contours.image, log_level=log_level)
            if np.any(np.isinf(np.array(result.top_face_matched_scores))):
                if log_level >= LOG_LEVEL_WARN:
                    print("WARNING: unable to score reverse top face match")

            result.adj_face_matches, result.adj_face_scores = self.compare_adjacent_faces(image_warped, log_level=log_level)

            if self.match_result_settings.is_rotationally_symmetric: #try the reverse
                initial_matches = result.adj_face_matches
                initial_scores = result.adj_face_scores
                initial_confidence = np.sum(self.get_match_result_confidences(result, log_level=log_level))

                rotation_warp = cv.getRotationMatrix2D((int(self.top_face_contours.centroid[0]),int(self.top_face_contours.centroid[1])),180,1.0)
                affine3 = np.vstack([result.top_face_match.affine_warp, [0, 0, 1]])
                rotation3 = np.vstack([rotation_warp,[0, 0, 1]])
                combined3 = np.matmul( rotation3, affine3)
                affine_combined = combined3[0:2,:]
                image_combined = cv.warpAffine(image, affine_combined, (self.image.shape[1], self.image.shape[0]))

                result.adj_face_matches, result.adj_face_scores = self.compare_adjacent_faces(image_combined, log_level=log_level)
                rotated_confidence = np.sum(self.get_match_result_confidences(result, log_level=log_level))
                if initial_confidence > rotated_confidence:
                    result.adj_face_matches = initial_matches
                    result.adj_face_scores = initial_scores
                else:
                    result.top_face_match.affine_warp = affine_combined
                    result.top_face_match.affine_warp_components = AffineWarpComponents.decompose_affine_warp(result.top_face_match.affine_warp)
                    if log_level >= LOG_LEVEL_DEBUG:
                        plt.imshow(image_combined, cmap='gray')
                        plt.title("Reversed symmetric version: (%2.0f%% vs %2.0f%%)"%(initial_confidence*100,rotated_confidence*100))
                        plt.show()
                        print("DEBUG: reverse version of rotaionally symmetric die face used (%2.0f%% vs %2.0f%%)"%(initial_confidence*100,rotated_confidence*100))

        result.confidences = self.get_match_result_confidences(result, log_level=log_level)
        result.confidence = np.sum(result.confidences)
        return result
    
    
    def compare_adjacent_faces(self, image_warped, log_level=LOG_LEVEL_FATAL):
        adj_face_matches = list()
        adj_face_scores = list()
    
        for i in np.arange(0,len(self.adjacent_face_contours)):
            adjacent_result = self.adjacent_face_matchers[i].find_in_image(image_warped, self.adjacent_face_masks[i], allow_distortion=True, log_level=log_level)
            adj_face_matches.append(adjacent_result)

            if adjacent_result.affine_warp is None or adjacent_result.matches_used == 0:
                scores = (np.inf, np.inf)
                if log_level >= LOG_LEVEL_VERBOSE:
                    print("WARNING: Unable to match adjacent face #%u"%i)
            elif abs(adjacent_result.affine_warp_components.scale_x - 1) > 0.5 or abs(adjacent_result.affine_warp_components.scale_y - 1) > 0.5:
                scores = (np.inf, np.inf)
                if log_level >= LOG_LEVEL_VERBOSE:
                    print("WARNING: adjacent face #%u found with an out of bounds scale: [%f,%f]"%(i,adjacent_result.affine_warp_components.scale_x,adjacent_result.affine_warp_components.scale_y))
            else:
                adjacent_image_warped = cv.warpAffine(image_warped,adjacent_result.affine_warp, (self.image.shape[1], self.image.shape[0]))
                scores = self.adjacent_face_matchers[i].compare_contour_area(adjacent_image_warped,self.adjacent_face_masks[i], log_level=log_level)
            adj_face_scores.append(scores)
        return adj_face_matches, adj_face_scores

            
    def get_polar_mask(self, center, min_radius, max_radius, start_angle, stop_angle, log_level=LOG_LEVEL_FATAL):
        mask = np.zeros(self.image.shape,dtype=np.uint8)
        while start_angle > stop_angle:
            stop_angle += 360
        cv.ellipse(mask,center,(int(max_radius),int(max_radius)),0,start_angle,stop_angle,color=255,thickness=cv.FILLED)
        if min_radius > 0:
            cv.circle(mask,center,int(min_radius),0,thickness=cv.FILLED)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(mask)
            plt.title("Polar Mask: r=[%u,%u] ; a=[%u, %u]"%(min_radius,max_radius,start_angle,stop_angle))
            plt.show()
        return mask

    # def cartesian_to_polar(self,point_x_y):
    #     point_center = point_x_y - self.center
    #     radius = np.linalg.norm(point_center)
    #     angle = np.degrees(np.arctan2(point_center[1],point_center[0]))
    #     return np.array([radius,angle])
        

    # def polar_to_cartesian(self,point_r_t):
    #     angle_rad = np.deg2rad(point_r_t[1])
    #     point_center = np.array([point_r_t[0]*np.cos(angle_rad), point_r_t[1]*np.sin(angle_rad)])
    #     return self.center + point_center

    def get_match_result_confidences(self, match_result:DieFaceMatchResult, log_level=LOG_LEVEL_FATAL):
        confidence = np.zeros(2)
        score = (max(match_result.top_face_scores[0], match_result.top_face_matched_scores[0]),max(match_result.top_face_scores[0], match_result.top_face_matched_scores[0]))
        confidence[0] = max(0,(1-self.match_result_settings.top_face_scale*max(0,score[0]-self.match_result_settings.top_face_offset))*self.match_result_settings.top_face_weight)
        confidence[1] = max(0,(1-self.match_result_settings.top_face_scale*max(0,score[1]-self.match_result_settings.top_face_eroded_offset))*self.match_result_settings.top_face_eroded_weight)
        total_weight = self.match_result_settings.top_face_weight + self.match_result_settings.top_face_eroded_weight
        
        num_adj_faces = self.geometry.adjacent_faces
        if num_adj_faces > 0:
            total_weight += self.match_result_settings.adj_face_weight + self.match_result_settings.adj_face_eroded_weight
            num_face_scores_to_keep = int(num_adj_faces/2)+1
            individual_weights = (self.match_result_settings.adj_face_weight/num_face_scores_to_keep, self.match_result_settings.adj_face_eroded_weight/num_face_scores_to_keep)
            for i in np.arange(0,self.geometry.adjacent_faces): # score in match_result.adj_face_scores:
                score = match_result.adj_face_scores[i] if i < len(match_result.adj_face_scores) else (0,0)
                confidence = np.append(confidence,max(0,(1-self.match_result_settings.adj_face_scale*max(0,score[0]-self.match_result_settings.adj_face_offset))*individual_weights[0]))
                confidence = np.append(confidence,max(0,(1-self.match_result_settings.adj_face_scale*max(0,score[1]-self.match_result_settings.adj_face_eroded_offset))*individual_weights[1]))
            for i in np.arange(0,num_adj_faces-num_face_scores_to_keep):
                confidence_by_face = confidence[2::2] + confidence[3::2]
                worst_face = np.argmin(confidence_by_face)
                confidence = np.delete(confidence,[(worst_face+1)*2,(worst_face+1)*2+1])
        if total_weight > 0:
            confidence = confidence / total_weight
        return confidence

    
        

    def view(self):
        plt.subplot(1,3,1)
        image_RGB0 = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
        cv.circle(image_RGB0,self.center,int(self.circumscribed_radius*self.geometry.enscribed_face_ratio),color=(255,0,0),thickness=3)
        cv.circle(image_RGB0,self.center,int(self.circumscribed_radius*self.geometry.top_face_ratio),color=(0,255,0),thickness=3)
        cv.circle(image_RGB0,self.center,int(self.circumscribed_radius*self.geometry.adjacent_face_ratio),color=(0,0,255),thickness=3)
        plt.imshow(image_RGB0)
        plt.title("Geometry")
        plt.subplot(1,3,2)
        image_RGB = cv.cvtColor(self.top_face_contours.image,cv.COLOR_GRAY2RGB)
        channel = 0
        for a in self.adjacent_face_contours:
            image_RGB[:,:,channel] = cv.bitwise_or(image_RGB[:,:,channel],a.image)
            channel = np.mod(channel+1,3)
        plt.imshow(image_RGB)
        plt.title("Match Groups")
        plt.subplot(1,3,3)
        image_RGB2 = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)
        image_RGB2[:,:,0] = cv.bitwise_or(image_RGB2[:,:,0],self.top_face_mask)
        image_RGB2[:,:,1] = cv.bitwise_or(image_RGB2[:,:,1],self.top_face_mask)
        channel = 0
        for a in self.adjacent_face_masks:
            image_RGB2[:,:,channel] = cv.bitwise_or(image_RGB2[:,:,channel],a)
            channel = np.mod(channel+1,3)
        plt.imshow(image_RGB2)
        plt.title("Mask Groups")
        plt.gcf().set_size_inches(10,6)
        plt.show()




    @staticmethod
    def create_from_sample(sample, imaging_settings, match_settings=None, log_level=LOG_LEVEL_FATAL):
        image = sample.get_keypoint_detection_image(imaging_settings, log_level=log_level)
        return DieFaceTemplate(image, sample.geometry, sample.circumscribed_radius, match_settings, log_level=log_level)