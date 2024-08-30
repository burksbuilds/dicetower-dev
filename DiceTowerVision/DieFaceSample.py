import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from DiceTowerVision.DiceTowerTools import *
from scipy.spatial.transform import Rotation
import math
import mahotas
from itertools import compress

class DieFaceSample:

    edge_opening_kernel_size = 3
    crop_padding = 25
    MIN_NUMBER_CONTOUR_SIZE = 25 #driving factor for min area is 6vs9 dots on d12

    def __init__(self, image_cropped_RGBA, geometry, imaging_settings=None, log_level=LOG_LEVEL_FATAL):
        if image_cropped_RGBA.shape[2] != 4:
            raise TypeError("Image must contain 4 channels: RGBA")
        self.image = image_cropped_RGBA
        self.image_hls = cv.cvtColor(self.image[:,:,0:3],cv.COLOR_RGB2HLS)
        self.geometry = geometry
        self.center = (self.image.shape[1]//2, self.image.shape[0]//2)
        self.circumscribed_radius = min(self.center) - DieFaceSample.crop_padding
        self.detector = cv.ORB.create() #cv.SIFT.create()  #cv.AKAZE.create(threshold=0.02)
        if imaging_settings is not None:
            self.update_keypoints(imaging_settings, log_level)
            self.update_moments(log_level=log_level)
    
    def get_circular_mask(self, circumscribe_ratio=1.0, log_level=LOG_LEVEL_FATAL):
        mask_radius = int(self.circumscribed_radius * circumscribe_ratio)
        
        mask = np.zeros((self.image.shape[0],self.image.shape[1]), dtype=np.uint8)
        mask = cv.circle(mask,self.center,mask_radius,255,cv.FILLED)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(mask, cmap='gray')
            plt.show()
        return mask
    
    def view_keypoints(self, keypoints=None, imaging_settings=None):
        if imaging_settings is None:
            image_debug = cv.cvtColor(self.image,cv.COLOR_RGBA2GRAY)
        else:
            image_debug = self.__get_keypoint_detection_image(imaging_settings)
        if keypoints is None:
            keypoints = self.keypoints
        image_debug = cv.drawKeypoints(image_debug,keypoints,0,(0,0,255),flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        plt.imshow(image_debug,cmap='gray')
        plt.title("Keypoints")
        plt.show()

    def view_geometry(self):
        image_debug = self.image[:,:,0:3].copy()
        cv.circle(image_debug,self.center,int(self.circumscribed_radius),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["enscribed_perimeter_ratio"]),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["circumscribed_face_ratio"]),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["enscribed_face_ratio"]),(255,255,255),3)
        cv.circle(image_debug,self.center,int(self.circumscribed_radius*self.geometry["pixel_comparison_ratio"]),(255,255,255),3)
        plt.imshow(image_debug)
        plt.title("Geometry")
        plt.show()

    def view_moments(self):
        print("Raw Moments:")
        for m in self.moments:
            print(m[1])
        print("Hu Moments:")
        for m in self.moments:
            print(np.array2string(m[2],max_line_width=32767))
        print("Zernike Moments:")
        for m in self.moments:
            print(np.array2string(m[3], max_line_width=32767))


    def __get_keypoint_detection_image(self, imaging_settings, log_level=LOG_LEVEL_FATAL):
        sample_mask = cv.morphologyEx(self.image[:,:,3],cv.MORPH_ERODE,get_kernel(5),iterations=3)
        hue_mask = get_hue_mask(self.image_hls[:,:,0],imaging_settings["hue_start"],imaging_settings["hue_end"])
        _, body_brightness_mask = get_brightness_masks(self.image_hls[:,:,1],imaging_settings["number_brightness"], imaging_settings["brightness_threshold"])
        numbers_mask = cv.bitwise_not(cv.bitwise_and(hue_mask, body_brightness_mask))    

        image_BW = cv.bitwise_and(numbers_mask,sample_mask) 
        image_BW = cv.morphologyEx(image_BW,cv.MORPH_OPEN,get_kernel(3),iterations=1)
        image_BW = cv.morphologyEx(image_BW,cv.MORPH_CLOSE,get_kernel(3),iterations=1)

        image_contours = draw_all_contours(image_BW,min_area=self.MIN_NUMBER_CONTOUR_SIZE,draw_style=cv.LINE_4)
        # contours, _ = cv.findContours(image_BW,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # major_contours = [c for c in contours if cv.contourArea(c) > self.MIN_NUMBER_CONTOUR_SIZE]
        # contour_mask = np.zeros(image_BW.shape, dtype=np.uint8)
        # cv.drawContours(contour_mask,major_contours,-1,255,thickness=cv.FILLED)
        # image_contours = cv.bitwise_and(image_BW, contour_mask)

        if log_level >= LOG_LEVEL_VERBOSE:
            plt.subplot(1,3,1)
            plt.imshow(numbers_mask, cmap='gray')
            plt.title("Unfiltered")
            plt.subplot(1,3,2)
            plt.imshow(image_BW, cmap='gray')
            plt.title("Filtered")
            plt.subplot(1,3,3)
            plt.imshow(image_contours, cmap='gray')
            plt.title("Contours")
            plt.suptitle("Keypoint Detection Image")
            plt.show();
        return image_contours


    def update_keypoints(self, imaging_settings, log_level=LOG_LEVEL_FATAL):
        mask = self.get_circular_mask()
        self.keypoint_image = self.__get_keypoint_detection_image(imaging_settings,log_level)
        self.keypoints, self.descriptors = self.detector.detectAndCompute(self.keypoint_image, mask)
        if log_level>=LOG_LEVEL_VERBOSE:
            self.view_keypoints(imaging_settings=imaging_settings)


    # cache central and hu moments for all contours
    def update_moments(self,log_level=LOG_LEVEL_FATAL):
        image_BW = self.keypoint_image
        contours, _ = cv.findContours(image_BW,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        all_raw_moments = [(c,self.get_moments_from_contours([c],log_level)) for c in contours]
        raw_moments = [m for m in all_raw_moments if m[1][0]["m00"] >= self.MIN_NUMBER_CONTOUR_SIZE] #prevent divide by zero
        raw_moments.sort(key= lambda m : m[1][0]["m00"],reverse=True)

        unscaled_hu_moments = [(m[0],m[1][0],cv.HuMoments(m[1][0]),m[1][1]) for m in raw_moments]
        scaled_hu_moments = [(m[0],m[1],normalize_hu_moments(m[2]), m[3]) for m in unscaled_hu_moments if np.all(m[2])]
        self.moments = scaled_hu_moments
        for i in np.arange(0,len(self.moments)):
            self.moments[i][2][6] = abs(self.moments[i][2][6]) #sign of the 7th hu moment indicates a mirror, but it seems inconsistent
        if log_level >= LOG_LEVEL_VERBOSE:
            self.view_moments()
            # print("Raw Moments:")
            # print(self.moments[1])
            # print("Hu Moments (log):")
            # print(self.moments[2])
            # print("Zernike Moments:")
            # print(self.moments[3])



    def get_moments_from_contours(self,contours,log_level=None):
        image_BW = self.keypoint_image
        contour_mask = np.zeros(image_BW.shape, dtype=np.uint8)
        cv.drawContours(contour_mask,contours,-1,255,thickness=cv.FILLED)
        contour_image = cv.bitwise_and(image_BW,contour_mask)
        classical_moments = cv.moments(contour_image,binaryImage=True)
        centroid = (classical_moments["m10"]/classical_moments["m00"],classical_moments["m01"]/classical_moments["m00"])
        max_distance_from_centroid = max([max([ math.sqrt((p[0][0]-centroid[0])**2 + (p[0][1]-centroid[1])**2) for p in c]) for c in contours])
        if log_level >= LOG_LEVEL_VERBOSE:
            print("Zernike moments calculated from centroid (%f,%f) with radius %f"%(centroid[0],centroid[1],max_distance_from_centroid))
        zernike_moments = mahotas.features.zernike_moments(image_BW,max_distance_from_centroid)
        return (classical_moments, zernike_moments)

    def get_moments_in_radius(self, radius_ratio=1, log_level=LOG_LEVEL_FATAL):
        center = (self.image.shape[1]/2,self.image.shape[0]/2)
        moments = list()
        for i in np.arange(0,len(self.moments)):
            M = self.moments[i][1]
            centroid = (M['m10']/M['m00'], M['m01']/M['m00'])
            distance = math.sqrt((center[0]-centroid[0])**2 + (center[1]-centroid[1])**2)
            if distance <= radius_ratio*self.circumscribed_radius:
                moments.append(self.moments[i])
        if log_level >= LOG_LEVEL_WARN and len(moments) == 0:
            print("WARNING: no contours were found within the specified radius")
        if log_level >= LOG_LEVEL_VERBOSE:
            image_debug = np.zeros((self.image.shape[0],self.image.shape[1],1),dtype=np.uint8)
            cv.drawContours(image_debug,[m[0] for m in moments],-1,(255),thickness=cv.FILLED)
            image_debug = cv.bitwise_and(image_debug, self.keypoint_image)
            plt.imshow(image_debug, cmap='gray')
            plt.title("Contours Within Specificed Radius")
            plt.show()
        return moments

    def get_keypoints_in_radius(self, radius_ratio=1, log_level=LOG_LEVEL_FATAL):
        keypoints = list()
        descriptors = list()
        center = (self.image.shape[1]/2,self.image.shape[0]/2)
        for i in np.arange(0,len(self.keypoints)):
            distance = math.sqrt((center[0]-self.keypoints[i].pt[0])**2 + (center[1]-self.keypoints[i].pt[1])**2)
            if distance <= radius_ratio*self.circumscribed_radius:
                keypoints.append(self.keypoints[i])
                descriptors.append(self.descriptors[i])
        return keypoints, np.array(descriptors)



    
    csv_fieldnames = ["source","die_name","die_position_x","die_position_y","die_position_confidence","face_value","sample_number","score","template_keypoints","sample_keypoints","total_matches","good_matches","used_matches","hu_moment_comparison","z_moment_comparison","raw_moment_projected_2d","raw_moment_projected_3d","hu_moment_projected_2d","hu_moment_projected_3d","weighted_pixel_template","weighted_pixel_differences","affine_pixel_differences","affine_angle","affine_scale","rotation_angle","translation_distance","projected_circularity","projected_scale","projected_offset_x","projected_offset_y"]

    def compare_to(self, other, camera_matrix=None, log_level=LOG_LEVEL_FATAL):
        results = dict()

        hu_moment_comparison, z_moment_comparison, matching_hu_contours, matching_z_contours = self.compare_moments_to(other,log_level)
        
        enscribed_keypoints, enscribed_descriptors = self.get_keypoints_in_radius(self.geometry["enscribed_perimeter_ratio"], log_level=log_level)

        #matcher = cv.FlannBasedMatcher(            
        #    indexParams=dict(algorithm = 1, trees=5),
        #    searchParams=dict(checks=50))  #https://stackoverflow.com/questions/62581171/how-to-implement-kaze-and-a-kaze-using-python-and-opencv
        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(other.descriptors, enscribed_descriptors, k=2)
        
        if log_level>=LOG_LEVEL_VERBOSE:
            all_matches = [m for (m,n) in matches]
            comparison = cv.drawMatches(other.image,other.keypoints,self.image,enscribed_keypoints,all_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(comparison)
            plt.title("All found matches")
            plt.show()

        #weight_mask = cv.distanceTransform(self.get_circular_mask(0.5),cv.DIST_L2,5)
        enscribed_face_mask = self.get_circular_mask(self.geometry["enscribed_face_ratio"])
        enscribed_perimeter_mask = self.get_circular_mask(self.geometry["enscribed_perimeter_ratio"])
        pixel_comparison_mask = self.get_circular_mask(self.geometry["pixel_comparison_ratio"])
        #center_moment_mask = self.get_circular_mask(self.geometry["enscribed_face_ratio"])
        enscribed_moments = cv.moments(cv.bitwise_and(self.keypoint_image,enscribed_face_mask),binaryImage=True)
        normalized_enscribed_moments = normalize_hu_moments(np.array([v for k,v in enscribed_moments.items()]))
        enscribed_hu_moments = normalize_hu_moments(cv.HuMoments(enscribed_moments))

        results["template_sample"] = self
        results["template_keypoints"] = np.count_nonzero(self.keypoints)
        results["sample_keypoints"] = np.count_nonzero(other.keypoints)
        results["total_matches"] = np.count_nonzero(matches)
        results["weighted_pixel_template"] = np.sum(np.multiply(enscribed_face_mask,self.keypoint_image / 255))
        results["weighted_pixel_template_full"] = np.sum(np.multiply(pixel_comparison_mask,self.keypoint_image / 255))
        results["hu_moment_comparison"] = hu_moment_comparison
        results["z_moment_comparison"] = z_moment_comparison

        #will be updated later
        results["homography"] = None
        results["affine_warp"] = None
        results["used_matches"] = 0
        results["good_matches"] = 0
        results["rotation_angle"] = 0
        results["affine_angle"] = 0
        results["affine_scale"] = 0
        results["off_axis_angle"] = 360
        results["translation_distance"] = 0
        results["weighted_pixel_differences"] = results["weighted_pixel_template"]
        results["affine_pixel_differences"] = results["weighted_pixel_template"]
        results["affine_pixel_differences_full"] = results["weighted_pixel_template_full"]
        results["affine_pixel_differences_eroded"] = results["weighted_pixel_template"]
        results["affine_pixel_differences_eroded_full"] = results["weighted_pixel_template_full"]
        results["projected_circularity"] = 0
        results["projected_offset"] = math.sqrt(self.center[0]**2 + self.center[1]**2)
        results["projected_scale"] = 0
        results["projected_offset_x"] = 0
        results["projected_offset_y"] = 0
        results["raw_moment_projected_2d"] = 0
        results["raw_moment_projected_3d"] = 0
        results["hu_moment_projected_2d"] = 0
        results["hu_moment_projected_3d"] = 0

        
        
        homography = None
        match_ratio = 0.75
            
        results["good_match_ratio"] = match_ratio
        good_matches = [m for (m,n) in matches if m.distance < match_ratio*n.distance]
        results["good_matches"] = len(good_matches)

        if len(good_matches) < 4:
            return results
        
        if log_level>=LOG_LEVEL_VERBOSE:
            comparison = cv.drawMatches(other.image,other.keypoints,self.image,enscribed_keypoints,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(comparison)
            plt.title("Good Matches (distance < %1.1f)"%match_ratio)
            plt.show()

        other_points = np.float32([ other.keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        self_points = np.float32([ enscribed_keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

        affine_warp, affine_inlier_mask = cv.estimateAffinePartial2D(other_points,self_points)

        if affine_warp is not None:
            results["affine_warp"] = affine_warp
            results["affine_angle"] = np.degrees(np.arctan2(affine_warp[1,0],affine_warp[1,1]))
            results["affine_scale"] = math.sqrt(affine_warp[1,0]**2 + affine_warp[1,1]**2)
            if log_level >= LOG_LEVEL_VERBOSE:
                comparison = cv.drawMatches(other.image,other.keypoints,self.image,enscribed_keypoints,list(compress(good_matches,affine_inlier_mask)), None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(comparison)
                plt.title("Matches used in Affine Estimation")
                plt.show()
                print("Affine Warp Angle Estimate: %f ; Scale Estimate: %f"%(results["affine_angle"],results["affine_scale"]))
                corrected_image = cv.warpAffine(other.image, affine_warp, (self.image.shape[1],self.image.shape[0]))
                plt.imshow(corrected_image)
                plt.title("2D Warped Color Image of Sample to Match Template")
                plt.show()

            other_BW = cv.warpAffine(other.keypoint_image,affine_warp,(self.keypoint_image.shape[1], self.keypoint_image.shape[0]))
        
            if log_level>=LOG_LEVEL_DEBUG:
                diff_RGB = np.zeros((self.image.shape[1], self.image.shape[0],4), np.uint8)
                diff_RGB[:,:,1] = self.keypoint_image
                diff_RGB[:,:,2] = other_BW
                diff_RGB[:,:,3] = cv.bitwise_or(enscribed_perimeter_mask,pixel_comparison_mask) #alpha channel is just where keypoints areplt.imshow(diff_RGB)
                plt.imshow(diff_RGB)
                plt.title("[2D] Overlay of sample and template simplified features")
                plt.show()
        
            image_diff = cv.bitwise_xor(other_BW,self.keypoint_image )
            results["affine_pixel_differences"] = np.sum(np.multiply(enscribed_face_mask, image_diff/255))
            results["affine_pixel_differences_full"] = np.sum(np.multiply(pixel_comparison_mask, image_diff/255))
            results["affine_pixel_differences_eroded"] = np.sum(np.multiply(enscribed_face_mask, cv.morphologyEx(image_diff,cv.MORPH_ERODE,get_kernel(3),iterations=1)/255))
            results["affine_pixel_differences_eroded_full"] = np.sum(np.multiply(pixel_comparison_mask, cv.morphologyEx(image_diff,cv.MORPH_ERODE,get_kernel(3),iterations=1)/255))
            projected_moments_2d = cv.moments(np.bitwise_and(enscribed_face_mask,other_BW),binaryImage=True) 
            projected_hu_moments_2d = normalize_hu_moments(cv.HuMoments(projected_moments_2d))
            normalzied_moments_2d = np.array([v for k,v in projected_moments_2d.items()])
            results["raw_moment_projected_2d"] = score_vector_similarity(normalized_enscribed_moments,normalzied_moments_2d)
            results["hu_moment_projected_2d"] = score_vector_similarity(enscribed_hu_moments,projected_hu_moments_2d)


        affine_skew_warp, affine_skew_inlier_mask = cv.estimateAffine2D(other_points, self_points)

        if affine_skew_warp is not None: 
            #https://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation
            results["affine_skew_warp"] = affine_skew_warp
            angle = np.arctan2(affine_skew_warp[1,0],affine_skew_warp[1,1])
            results["affine_skew_angle"] = np.degrees(angle)
            results["affine_skew_scalex"] = math.sqrt(affine_skew_warp[1,0]**2 + affine_skew_warp[1,1]**2)
            shear_skewy = affine_skew_warp[0,1]*math.cos(angle) + affine_skew_warp[1,1]*math.sin(angle)
            if abs(math.sin(angle)) > 0.0001:
                results["affine_skew_scaley"] = (shear_skewy*math.cos(angle) - affine_skew_warp[0,1]) / math.sin(angle)
            else: 
                results["affine_skew_scaley"] = (affine_skew_warp[1,1] - shear_skewy*math.sin(angle)) / math.cos(angle)
            results["affine_skew_shear"] = shear_skewy/results["affine_skew_scaley"]
            if log_level >= LOG_LEVEL_VERBOSE:
                comparison = cv.drawMatches(other.image,other.keypoints,self.image,enscribed_keypoints,list(compress(good_matches,affine_skew_inlier_mask)), None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(comparison)
                plt.title("Matches used in Affine Skew Estimation")
                plt.show()
                print("Affine Skew Warp Angle Estimate: %f ; Scale Estimate: (%f,%f) ; Shear Estimate: %f"%(results["affine_skew_angle"],results["affine_skew_scalex"],results["affine_skew_scaley"],results["affine_skew_shear"]))
                corrected_image = cv.warpAffine(other.image, affine_skew_warp, (self.image.shape[1],self.image.shape[0]))
                plt.imshow(corrected_image)
                plt.title("2D Skewed Warped Color Image of Sample to Match Template")
                plt.show()

            other_BW = cv.warpAffine(other.keypoint_image,affine_skew_warp,(self.keypoint_image.shape[1], self.keypoint_image.shape[0]))
        
            if log_level>=LOG_LEVEL_DEBUG:
                diff_RGB = np.zeros((self.image.shape[1], self.image.shape[0],4), np.uint8)
                diff_RGB[:,:,1] = self.keypoint_image
                diff_RGB[:,:,2] = other_BW
                diff_RGB[:,:,3] = cv.bitwise_or(enscribed_perimeter_mask,pixel_comparison_mask) #alpha channel is just where keypoints areplt.imshow(diff_RGB)
                plt.imshow(diff_RGB)
                plt.title("[2DS] Overlay of sample and template simplified features")
                plt.show()
        
            results["affine_skew_pixel_differences"] = np.sum(np.multiply(enscribed_face_mask, cv.bitwise_xor(other_BW,self.keypoint_image )/255))
            projected_moments_2ds = cv.moments(np.bitwise_and(enscribed_face_mask,other_BW),binaryImage=True) 
            projected_hu_moments_2ds = normalize_hu_moments(cv.HuMoments(projected_moments_2ds))
            normalzied_moments_2ds = np.array([v for k,v in projected_moments_2ds.items()])
            results["raw_moment_projected_2ds"] = score_vector_similarity(normalized_enscribed_moments,normalzied_moments_2ds)
            results["hu_moment_projected_2ds"] = score_vector_similarity(enscribed_hu_moments,projected_hu_moments_2ds)


        homography, homography_inlier_mask = cv.findHomography(other_points, self_points, cv.RANSAC, 3.0)

        if homography is None:
            return results
        

        if log_level>=LOG_LEVEL_VERBOSE:
            comparison = cv.drawMatches(other.image,other.keypoints,self.image,enscribed_keypoints,list(compress(good_matches,homography_inlier_mask)), None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(comparison)
            plt.title("Matches used in Homography Estimation")
            plt.show()
            corrected_image = cv.warpPerspective(other.image,homography,(other.image.shape[1],other.image.shape[0]))
            plt.imshow(corrected_image)
            plt.title("3D Warped Color Image of Sample to Match Template")
            plt.show()
            
        results["homography"] = homography
        results["used_matches"] = np.count_nonzero(homography_inlier_mask)
            
        
        if log_level >= LOG_LEVEL_VERBOSE:
            print("using good match ratio of %1.1f"%results["good_match_ratio"])



        other_BW = cv.warpPerspective(other.keypoint_image,homography,(self.keypoint_image.shape[1], self.keypoint_image.shape[0]))
        
        if log_level>=LOG_LEVEL_DEBUG:
            diff_RGB = np.zeros((self.image.shape[1], self.image.shape[0],4), np.uint8)
            diff_RGB[:,:,1] = self.keypoint_image
            diff_RGB[:,:,2] = other_BW
            diff_RGB[:,:,3] = cv.bitwise_or(enscribed_perimeter_mask,pixel_comparison_mask) #alpha channel is just where keypoints areplt.imshow(diff_RGB)
            plt.imshow(diff_RGB)
            plt.title("[3D] Overlay of sample and template simplified features")
            plt.show()
        
        results["weighted_pixel_differences"] = np.sum(np.multiply(enscribed_face_mask, cv.bitwise_xor(other_BW,self.keypoint_image )/255))
        projected_moments_3d = cv.moments(np.bitwise_and(enscribed_face_mask,other_BW),binaryImage=True) 
        normalzied_moments_3d = normalize_hu_moments(np.array([v for k,v in projected_moments_3d.items()]))
        projected_hu_moments_3d = normalize_hu_moments(cv.HuMoments(projected_moments_3d))
        results["raw_moment_projected_3d"] = score_vector_similarity(normalized_enscribed_moments,normalzied_moments_3d)
        results["hu_moment_projected_3d"] = score_vector_similarity(enscribed_hu_moments,projected_hu_moments_3d)
        
        test_scale = 0.25
        test_circle = self.get_circular_mask(test_scale)
        test_circle_transform = cv.warpPerspective(test_circle, homography, test_circle.shape)
        test_contours, _ = cv.findContours(test_circle_transform, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)



        if len(test_contours) == 1:
            if len(test_contours[0]) >= 5:
                (x,y), (major,minor),angle = cv.fitEllipse(test_contours[0])
                results["projected_offset_x"] = self.center[0]-x
                results["projected_offset_y"] = self.center[1]-y
                results["projected_offset"] = math.sqrt((self.center[0]-x)**2 + (self.center[1]-y)**2) / self.circumscribed_radius
                results["projected_circularity"] = major**2 / minor**2
                results["projected_scale"] = (minor*major/4)/((self.circumscribed_radius*test_scale)**2)
                
                if log_level >= LOG_LEVEL_VERBOSE:
                    print("Offset=%f ; circularity=%f ; scale=%f"%(results["projected_offset"],results["projected_circularity"],results["projected_scale"]))
            else:
                if log_level >= LOG_LEVEL_DEBUG:
                    print("Not enough points to fit an ellipse")
        else:
            if log_level >= LOG_LEVEL_DEBUG:
                print("%u contours were found in transformed test circle image (not just 1!)"%len(test_contours))

        if log_level >= LOG_LEVEL_VERBOSE:
            diff_RGB = np.zeros((test_circle.shape[1], test_circle.shape[0],3), np.uint8)
            diff_RGB[:,:,1] = test_circle
            diff_RGB[:,:,2] = test_circle_transform
            plt.imshow(diff_RGB)
            plt.title("Overlay of test circle and transformed test circle")
            plt.show()

        if camera_matrix is None:
            if log_level >= LOG_LEVEL_WARN:
                print("Unable to decompose homography because no camera matrix was provided")
            return results
        
        solutions, rotations, translations, normals = cv.decomposeHomographyMat(homography,camera_matrix)


        for i in np.arange(0,solutions):
            r = Rotation.from_matrix(rotations[i])
            angles = r.as_euler("zyx",degrees=True)
            distances = translations[i]
            off_axis_angle = math.sqrt(angles[1]**2 + angles[2]**2)
            translation_distance = math.sqrt(distances[0]**2 + distances[1]**2 + distances[2]**2)
            # if log_level >= LOG_LEVEL_VERBOSE:
            #     print(angles)
            #     print(distances)
            #     print(normals[i])
            if translation_distance < 100 and off_axis_angle < results["off_axis_angle"]:
                results["off_axis_angle"] = off_axis_angle
                results["translation_distance"] = translation_distance
                results["rotation_angle"] = angles[0]

        if off_axis_angle >= 360 and log_level>=LOG_LEVEL_DEBUG:
            print("Found homography required excessive translation distance or off-axis angle")

        if log_level >= LOG_LEVEL_VERBOSE:
            print("rotation=%f ; distance=%f ; off-axis=%f"%(results["rotation_angle"], results["translation_distance"] ,results["off_axis_angle"]))

        test_score = np.array([1/r["hu_moment_projected_2d"]  for r in [results]])
        return results

    
    def compare_moments_to(self, other, log_level=LOG_LEVEL_FATAL):
        #from largest area to smallest area, match each contour from this image to its best match in the other image
        these_moments = self.get_moments_in_radius(self.geometry["enscribed_face_ratio"], log_level)
        other_moments = other.get_moments_in_radius(1, log_level)
        #these_moments_order = np.argsort(np.array([m["m00"] for m in these_moments[0]]))
        best_matching_hu_contours = list()
        best_matching_z_contours = list()
        #best_matching_contour_score = np.zeros(len(these_moments))
        weighted_hu_score = 0
        weighted_z_score = 0

        for i in np.arange(0,len(these_moments)):
            other_contour_scores_h = np.zeros(len(other_moments))
            other_contour_scores_z = np.zeros(len(other_moments))
            for j in np.arange(0,len(other_contour_scores_h)):
                other_contour_scores_h[j] = score_vector_similarity(these_moments[i][2], other_moments[j][2])
                other_contour_scores_z[j] = score_vector_similarity(these_moments[i][3], other_moments[j][3])
                if log_level >= LOG_LEVEL_VERBOSE:
                    print("[H] This contour #%u vs that contour #%u : %f"%(i,j,other_contour_scores_h[j]))
                    print("[Z] This contour #%u vs that contour #%u : %f"%(i,j,other_contour_scores_z[j]))
            best_matching_hu_contour_index = np.argmin(other_contour_scores_h)
            best_matching_z_contour_index = np.argmin(other_contour_scores_z)
            hu_score = other_contour_scores_h[best_matching_hu_contour_index]
            z_score = other_contour_scores_z[best_matching_z_contour_index]
            best_matching_hu_contours.append((these_moments[i],other_moments[best_matching_hu_contour_index],hu_score))
            best_matching_z_contours.append((these_moments[i],other_moments[best_matching_z_contour_index],z_score))
            weighted_hu_score = weighted_hu_score + hu_score
            weighted_z_score = weighted_z_score + z_score
        return weighted_hu_score / len(these_moments), weighted_z_score / len(these_moments), best_matching_hu_contours, best_matching_z_contours

 

    @staticmethod
    def crop_sample_from_image(image_RGB, mask, center, radius, log_level=LOG_LEVEL_DEBUG):
        cropX1 = max(center[0]-radius-DieFaceSample.crop_padding,0)
        cropX2 = min(center[0]+radius+DieFaceSample.crop_padding,image_RGB.shape[1]-1)
        cropY1 = max(center[1]-radius-DieFaceSample.crop_padding,0)
        cropY2 = min(center[1]+radius+DieFaceSample.crop_padding,image_RGB.shape[0]-1)
        image_Crop_RGBA = cv.cvtColor(image_RGB[cropY1:cropY2,cropX1:cropX2,:],cv.COLOR_RGB2RGBA)
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.imshow(image_Crop_RGBA)
            plt.title("Cropped RGB")
            plt.show()
        if mask is not None:
            image_Crop_RGBA[:,:,3] = mask[cropY1:cropY2,cropX1:cropX2]
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.imshow(image_Crop_RGBA[:,:,3],cmap='gray')
            plt.title("Cropped Alpha Mask")
            plt.show()
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(image_Crop_RGBA)
            plt.title("Cropped and Masked")
            plt.show()
        return image_Crop_RGBA
    
    @staticmethod
    def __crop_and_mask_template_image(image_RGB, roi_mask=None, log_level=LOG_LEVEL_FATAL):
    #1) Threshold saturation to get a grayscale image that rejects shadows and highlight the one die we expect near the middle
        image_HLS = cv.cvtColor(image_RGB,cv.COLOR_RGB2HLS)
        if log_level >= LOG_LEVEL_VERBOSE:
            plt.imshow(image_RGB)
            plt.title("Original Template Image")
            plt.show()
            plt.imshow(image_HLS[:,:,0],cmap='gray')
            plt.title("Hue")
            plt.show()
            plt.imshow(image_HLS[:,:,2],cmap='gray')
            plt.title("Saturation")
            plt.show()
            plt.imshow(image_HLS[:,:,1],cmap='gray')
            plt.title("Brightness")
            plt.show()

        image_Gray = image_HLS[:,:,1] #use brightness channel because the BG is just so dark
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(image_Gray,cmap='gray')
            plt.title("Grayscale Image (Brightness)")
            plt.show()
        #_,image_BW = cv.threshold(image_Gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        _,image_BW = cv.threshold(image_Gray,50,255,cv.THRESH_BINARY)

        if roi_mask is not None:
            image_BW[np.where(roi_mask == 0)] = 0 #only look within the region of interest mask

        #2) reject noise with an opening morphology (erode then dilate)
        image_Filtered_BW = image_BW.copy()
        image_Filtered_BW = cv.morphologyEx(image_Filtered_BW,cv.MORPH_OPEN,get_kernel(DieFaceSample.edge_opening_kernel_size),iterations=2)

        #3) find the biggest contour and create a convex hull around it. This is the die.
        contours, _ = cv.findContours(image_Filtered_BW,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if log_level>=LOG_LEVEL_DEBUG:
            plt.imshow(image_BW,cmap='gray')
            plt.title("Threshold on Grayscale")
            plt.show()
            plt.imshow(image_Filtered_BW,cmap='gray')
            plt.title("Filtered Matching Areas")
            plt.show()
        if len(contours) < 1:
            if log_level >= LOG_LEVEL_ERROR:
                plt.imshow(image_Filtered_BW,cmap='gray')
                plt.show()
            raise Exception("Error cropping template image: No contour found")
        contour_properties = [(cv.contourArea(c),cv.minAreaRect(c), cv.convexHull(c)) for c in contours]
        contour_Areas = [cp[0] if (cp[1][1][0]/cp[1][1][1] < 3 and cp[1][1][1]/cp[1][1][0] < 3 and cv.contourArea(cp[2])/cp[0] < 2) else 0 for cp in contour_properties]
        hull = cv.convexHull(contours[contour_Areas.index(max(contour_Areas))])
        image_Mask_BW = np.zeros(image_Gray.shape, dtype=np.uint8)
        cv.drawContours(image_Mask_BW, [hull], 0, 255,thickness=cv.FILLED)

        #4) circumscribe the convext hull to get the center and rotation-invariant "extents" of the die
        (x,y), r = cv.minEnclosingCircle(hull)
        center = (int(x),int(y))
        radius = int(r)
        if log_level>=LOG_LEVEL_VERBOSE:
            plt.imshow(image_Mask_BW,cmap='gray')
            plt.title("Mask used to extract die from template")
            plt.show()
            image_ID = image_RGB.copy()
            cv.drawContours(image_ID,[hull],0,(255,255,255),3)
            cv.circle(image_ID,center,radius,(255,0,0),3)
            plt.imshow(image_ID)
            plt.title("Dice overlaid with contour and circumscribed boundary")
            plt.show()

        #5) crop around the circle (with padding) to get a scale-invariant image of the die. move the mask into the alpha channel
        
        return DieFaceSample.crop_sample_from_image(image_RGB, image_Mask_BW,center,radius,log_level)

    @staticmethod
    def __create_from_uncropped_image(image_or_file, geometry, imaging_settings=None, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        if not isinstance(image_or_file, np.ndarray):
            image_BGR = cv.imread(image_or_file)
            if image_BGR is None:
                if log_level>=LOG_LEVEL_ERROR:
                    print("ERROR: FILE NOT FOUND! Verify that relative paths join properly")
                    print(image_or_file)
                    print(os.getcwd())
                raise FileNotFoundError()
            image_RGB = cv.cvtColor(image_BGR,cv.COLOR_BGR2RGB)
        else:
            image_RGB = image_or_file
            if image_RGB.shape[2] != 3:
                raise Exception("Image must contain at least three channels")
        return DieFaceSample(DieFaceSample.__crop_and_mask_template_image(image_RGB, roi_mask, log_level), geometry, imaging_settings, log_level=log_level)
    
    @staticmethod
    def __create_from_cropped_image(image_or_file, geometry, imaging_settings=None, log_level=LOG_LEVEL_FATAL):
        if not isinstance(image_or_file, np.ndarray):
            image_raw = cv.imread(image_or_file,cv.IMREAD_UNCHANGED)
            if image_raw is None:
                if log_level>=LOG_LEVEL_ERROR:
                    print("ERROR: FILE NOT FOUND! Verify that relative paths join properly")
                    print(image_or_file)
                    print(os.getcwd())
                raise FileNotFoundError()
        else:
            image_raw = image_or_file
        if image_raw.shape[2] < 3:
            raise Exception("Image must contain RGB channels, optionally Alpha")
        if image_raw.shape[2] == 3:
            return DieFaceSample(cv.cvtColor(image_raw,cv.COLOR_BGR2RGBA), geometry, imaging_settings, log_level)
        if image_raw.shape[2] == 4:
            return DieFaceSample(cv.cvtColor(image_raw,cv.COLOR_BGRA2RGBA), geometry, imaging_settings, log_level)
        else:
            raise Exception("Image must contain RGB channels, optionally Alpha")
        
    @staticmethod
    def create_from_image(image_or_file, geometry, imaging_settings=None, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        if autocrop:
            return DieFaceSample.__create_from_uncropped_image(image_or_file, geometry, imaging_settings, roi_mask, log_level)
        else:
            return DieFaceSample.__create_from_cropped_image(image_or_file, geometry, imaging_settings, log_level)
        
    @staticmethod
    def create_from_image_files(image_files, geometry, imaging_settings=None, autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        if os.path.isdir(image_files):
            dir_files = [f for f in os.listdir(image_files) if os.path.isfile(os.path.join(image_files,f))]
            return [DieFaceSample.create_from_image(f, geometry, imaging_settings,  autocrop, roi_mask, log_level) for f in dir_files]
        else:
            return [DieFaceSample.create_from_image(f, geometry, imaging_settings,  autocrop, roi_mask, log_level) for f in image_files]
        

    def create_from_image_file_template(image_file_template, geometry, imaging_settings=None,autocrop=True, roi_mask=None, log_level=LOG_LEVEL_FATAL):
        images_RGB = get_mactching_images_RGB(image_file_template, log_level)
        samples = [DieFaceSample.create_from_image(i, geometry, imaging_settings, autocrop, roi_mask, log_level) for i in images_RGB]
        return samples

