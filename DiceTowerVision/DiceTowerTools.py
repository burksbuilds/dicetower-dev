import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import math

LOG_LEVEL_SILENT = 0
LOG_LEVEL_FATAL = 1
LOG_LEVEL_ERROR = 2
LOG_LEVEL_WARN = 3
LOG_LEVEL_INFO = 4
LOG_LEVEL_DEBUG = 5
LOG_LEVEL_VERBOSE = 6


#HUE_MASK_BUFFER = 5

def get_kernel(size):
    return np.ones((size,size),np.uint8)

def get_hue_mask(image_hues, hue_start, hue_end, margin=0):
    hue_start = int(np.mod(hue_start-margin,180))
    hue_end = int(np.mod(hue_end+margin,180))
    if hue_start > hue_end:
        return cv.bitwise_or(cv.inRange(image_hues,0,hue_end),cv.inRange(image_hues,hue_start,180))
    else:
        return cv.inRange(image_hues,hue_start,hue_end)
    
def get_brightness_masks(image_brightness, number_brightness, brightness_threshold, filter=9):
    brightness_filtered = cv.GaussianBlur(image_brightness,(filter,filter),0)
    if number_brightness < brightness_threshold: #black numbers
        number_mask = cv.inRange(brightness_filtered,0,number_brightness)
        body_mask = cv.inRange(brightness_filtered,brightness_threshold,255)
    else: #white numbers
        number_mask = cv.inRange(brightness_filtered,number_brightness,255)
        body_mask = cv.inRange(brightness_filtered,0,brightness_threshold)
    return number_mask, body_mask

def draw_all_contours(image_BW, fill_threshold=0.0, min_area=0, draw_style=cv.FILLED, log_level=LOG_LEVEL_FATAL):
    contours, _ = cv.findContours(image_BW,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    image_filled = np.zeros(image_BW.shape,dtype=np.uint8)
    for contour in contours:
        mask = np.zeros(image_BW.shape,dtype=np.uint8)
        cv.drawContours(mask,[contour],0,255,cv.FILLED)
        mask_size = cv.contourArea(contour)
        if mask_size > min_area:
            fill_ratio = np.count_nonzero(cv.bitwise_and(mask,image_BW)) / mask_size
            if fill_ratio >= fill_threshold:
                if draw_style == cv.FILLED:
                    image_filled = np.bitwise_or(image_filled,mask)
                else:
                    image_filled = np.bitwise_or(image_filled,np.bitwise_and(mask,image_BW))
    if log_level >= LOG_LEVEL_VERBOSE:
        plt.imshow(image_filled, cmap='gray')
        plt.title("Image With All Contours Drawn")
        plt.show()
    return image_filled


def get_sample_offset(sample,template, warp):
    #warp is transformation from sample -> template.
    #we want transformation from template -> sample,
    # then put the center point of the template into the transformation and compare it to the centerpoint of the sample. this is the offset.
    reverse_warp = np.zeros((2,3))
    tp = np.transpose(warp[0:2,0:2])
    reverse_warp[0:2,0:2] = tp
    reverse_warp[0:2,2] = -1 * np.matmul(tp,warp[0:2,2])
    template_center_on_sample = np.matmul(reverse_warp[0:2,0:2], np.array(template.center)) + reverse_warp[0:2,2]
    return template_center_on_sample - np.array(sample.center)
 

def view_face_match_scores(face_matches_list, context=""):
    sample_ids = np.arange(0,len(face_matches_list))
    scores = np.zeros((len(sample_ids),len(face_matches_list[0])))
    scores_eroded = np.zeros(scores.shape)
    for i, face_match_results in enumerate(face_matches_list):
        sorted_results = sorted(face_match_results,key=lambda r: -1*r.confidence)
        scores[i,:] = np.array([r.top_face_scores[0] for r in sorted_results])
        scores_eroded[i,:] = np.array([r.top_face_scores[1] for r in sorted_results])
    plt.subplot(2,1,1)
    plt.plot(sample_ids,scores)
    plt.title("Top Face Scores")
    plt.ylim((0,10))
    plt.subplot(2,1,2)
    plt.plot(sample_ids,scores_eroded)
    plt.title("Top Face Scores (Eroded)")
    plt.ylim((0,10))
    plt.gcf().set_size_inches((10,6))
    plt.suptitle("Top Face Score Comparison (%s)"%context)
    plt.show()

    for a in np.arange(len(face_matches_list[0][0].adj_face_scores)):
        for i, face_match_results in enumerate(face_matches_list):
            sorted_results = sorted(face_match_results,key=lambda r: -1*r.confidence)
            scores[i,:] = np.array([r.adj_face_scores[a][0] for r in sorted_results])
            scores_eroded[i,:] = np.array([r.adj_face_scores[a][1] for r in sorted_results])
        plt.subplot(2,1,1)
        plt.plot(sample_ids,scores)
        plt.title("Adjacent Face Scores")
        plt.ylim((0,10))
        plt.subplot(2,1,2)
        plt.plot(sample_ids,scores_eroded)
        plt.title("Adjacent Face Scores (Eroded)")
        plt.ylim((0,10))
        plt.gcf().set_size_inches((10,6))
        plt.suptitle("Adjacent Face #%u Score Comparison (%s)"%(a,context))
        plt.show()


# def rescale_scores(scores, min_score=0, max_score=1):
#     scale = 1/(max_score-min_score)
#     offset = min_score*scale
#     return  np.array([max(min(score*scale - offset,1),0) for score in scores])

# #def filter_scores(scores, min_score=0, max_score=1):


# def get_scores_from_good_keypoints(results, log_level=LOG_LEVEL_FATAL):
#     good_match_keypoint_ratio = np.array([r["good_matches"]/r["template_keypoints"] for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),good_match_keypoint_ratio)
#         plt.ylim((0,1))
#         plt.title("Good Keypoint Matches")
#         plt.show()
#     return good_match_keypoint_ratio

# def get_scores_from_used_keypoints(results, log_level=LOG_LEVEL_FATAL):
#     used_match_keypoint_ratio = np.array([r["used_matches"]/r["template_keypoints"] for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),used_match_keypoint_ratio)
#         plt.ylim((0,1))
#         plt.title("Used Keypoint Matches")
#         plt.show()
#     return used_match_keypoint_ratio

# def get_scores_from_3d_pixel_alignment(results, log_level=LOG_LEVEL_FATAL):
#     non_matching_pixel_ratio = np.array([max(0,1 - r["weighted_pixel_differences"]/r["weighted_pixel_template"]) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),non_matching_pixel_ratio)
#         plt.ylim((0,1))
#         plt.title("3D Non-Matching Pixels")
#         plt.show()
#     return non_matching_pixel_ratio

# def get_scores_from_2d_pixel_alignment(results, log_level=LOG_LEVEL_FATAL):
#     non_matching_pixel_ratio = np.array([max(0,1 - r["affine_pixel_differences"]/r["weighted_pixel_template"]) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),non_matching_pixel_ratio)
#         plt.ylim((0,1))
#         plt.title("2D Non-Matching Pixels (Center)")
#         plt.show()
#     return non_matching_pixel_ratio

# def get_scores_from_2d_pixel_alignment_full(results, log_level=LOG_LEVEL_FATAL):
#     non_matching_pixel_ratio = np.array([max(0,1 - r["affine_pixel_differences_full"]/r["weighted_pixel_template_full"]) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),non_matching_pixel_ratio)
#         plt.ylim((0,1))
#         plt.title("2D Non-Matching Pixels (Full)")
#         plt.show()
#     return non_matching_pixel_ratio

# def get_scores_from_2d_pixel_alignment_eroded(results, log_level=LOG_LEVEL_FATAL):
#     non_matching_pixel_ratio = np.array([max(0,1 - r["affine_pixel_differences_eroded"]/r["weighted_pixel_template"]) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),non_matching_pixel_ratio)
#         plt.ylim((0,1))
#         plt.title("2D Non-Matching Pixels (Center Eroded)")
#         plt.show()
#     return non_matching_pixel_ratio

# def get_scores_from_2d_pixel_alignment_eroded_full(results, log_level=LOG_LEVEL_FATAL):
#     non_matching_pixel_ratio = np.array([max(0,1 - r["affine_pixel_differences_eroded_full"]/r["weighted_pixel_template_full"]) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),non_matching_pixel_ratio)
#         plt.ylim((0,1))
#         plt.title("2D Non-Matching Pixels (Full Eroded)")
#         plt.show()
#     return non_matching_pixel_ratio

# def get_scores_from_off_angles(results, log_level=LOG_LEVEL_FATAL):
#     angle_offset_ratio = np.array([max(0,1 - r["off_axis_angle"]/90) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),angle_offset_ratio)
#         plt.ylim((0,1))
#         plt.title("Off-Axis Angle Result (1/90 deg)")
#         plt.show()
#     return angle_offset_ratio

# def get_scores_from_offset(results, log_level=LOG_LEVEL_FATAL):
#     offset_ratio = np.array([max(0,1 - r["projected_offset"]) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),offset_ratio)
#         plt.ylim((0,1))
#         plt.title("Offset of transformed test circle")
#         plt.show()
#     return offset_ratio

# def get_scores_from_circularity(results, log_level=LOG_LEVEL_FATAL):
#     circularity_score = np.array([max(0,1 - abs(r["projected_circularity"]-1)) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),circularity_score)
#         plt.ylim((0,1))
#         plt.title("Circularity of transformed test circle")
#         plt.show()
#     return circularity_score

# def get_scores_from_scale(results, log_level=LOG_LEVEL_FATAL):
#     scale_score = np.array([max(0, 1 - abs(r["projected_scale"]-1)) for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),scale_score)
#         plt.ylim((0,1))
#         plt.title("Scale of transformed test circle")
#         plt.show()
#     return scale_score

# def get_score_from_vector_similarity(results, key, log_level=LOG_LEVEL_FATAL):
#     score = np.array([0 if r[key]==0 else 1/r[key]  for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),score)
#         #plt.ylim((0,1))
#         plt.title("Vector Similarity: %s"%key)
#         plt.show()
#     return score

# def get_scores_from_hu_moments(results, log_level=LOG_LEVEL_FATAL):
#     moment_score = np.array([1/r["hu_moment_comparison"]  for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),moment_score)
#         #plt.ylim((0,1))
#         plt.title("Hu Moments Distances of Matching Contours")
#         plt.show()
#     return moment_score

# def get_scores_from_z_moments(results, log_level=LOG_LEVEL_FATAL):
#     moment_score = np.array([1/r["z_moment_comparison"]  for r in results])
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(results)),moment_score)
#         #plt.ylim((0,1))
#         plt.title("Zernike Moments Distances of Matching Contours")
#         plt.show()
#     return moment_score

# def get_weighted_scores(scores, weights, log_level=LOG_LEVEL_FATAL):
#     weight_sum = np.sum(weights)
#     weighted_scores = np.zeros(scores[0].shape,dtype=np.float32)
#     for i in np.arange(0,len(weights)):
#         weighted_scores = np.add(weighted_scores, scores[i]*weights[i]/weight_sum)
#     if log_level >= LOG_LEVEL_DEBUG:
#         plt.bar(np.arange(0,len(weighted_scores)),weighted_scores)
#         plt.ylim((0,1))
#         plt.title("Total Weighted Face Match Scores")
#         plt.show()
#     return weighted_scores

# #def get_passing_results(results, log_level=LOG_LEVEL_FATAL):


# def get_scores_from_results(results, log_level=LOG_LEVEL_FATAL):
#     #good_keypoint_scores = rescale_scores(get_scores_from_good_keypoints(results,log_level),0.2,0.5)
#     #used_keypoint_scores = rescale_scores(get_scores_from_used_keypoints(results,log_level),0.1,0.25)
#     #pixel_scores_3d = rescale_scores(get_scores_from_3d_pixel_alignment(results,log_level),0,1)
#     pixel_scores_2d = rescale_scores(get_scores_from_2d_pixel_alignment(results,log_level),0,1)
#     pixel_scores_2df = rescale_scores(get_scores_from_2d_pixel_alignment_full(results,log_level),0,1)
#     pixel_scores_2de = rescale_scores(get_scores_from_2d_pixel_alignment_eroded(results,log_level),0,1)
#     pixel_scores_2def = rescale_scores(get_scores_from_2d_pixel_alignment_eroded_full(results,log_level),0,1)
#     #angle_scores = rescale_scores(get_scores_from_off_angles(results,log_level),0.75)
#     #offset_scores = rescale_scores(get_scores_from_offset(results,log_level),0.75)
#     #circularity_scores = rescale_scores(get_scores_from_circularity(results,log_level),0.75)
#     #scale_scores = rescale_scores(get_scores_from_scale(results,log_level),0.75)
#     #hu_moment_scores = rescale_scores(get_scores_from_hu_moments(results,log_level),0,1)
#     #z_moment_scores = rescale_scores(get_scores_from_z_moments(results,log_level),0,1)
#     #weighted_scores = get_weighted_scores([good_keypoint_scores,used_keypoint_scores,pixel_scores,angle_scores, offset_scores, circularity_scores, scale_scores, moment_scores],[0,0,1,0,0,0,0,1],log_level)
#     #rescale_scores(get_score_from_vector_similarity(results,"raw_moment_projected_2d",log_level),0,1)
#     #rescale_scores(get_score_from_vector_similarity(results,"hu_moment_projected_2d",log_level),0,1)
#     weighted_scores = get_weighted_scores([pixel_scores_2d, pixel_scores_2de, pixel_scores_2df, pixel_scores_2def],[1,1,1,1],log_level)
#     return weighted_scores

def get_roi_mask_from_bg_image(image_RGB, range=(0, 10), filter=25, dilation=0, log_level=LOG_LEVEL_FATAL):
    image_gray = cv.cvtColor(image_RGB, cv.COLOR_RGB2GRAY)
    image_blur_gray = cv.blur(image_gray,(filter,filter))
    image_BW = cv.inRange(image_blur_gray,range[0], range[1])
    mask = np.zeros(image_BW.shape,np.uint8)
    contours, _ = cv.findContours(image_BW,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    if contours:
        max_contour = max(contours, key=cv.contourArea)
        cv.drawContours(mask,[max_contour],0,255,cv.FILLED)
        mask = cv.morphologyEx(mask,cv.MORPH_DILATE,get_kernel(3),iterations=dilation)
    else:
        if log_level >= LOG_LEVEL_WARN:
            plt.imshow(image_RGB)
            plt.title("NO BLACK BACKGROUND FOUND IN IMAGE!")
            plt.show()
    if log_level >= LOG_LEVEL_VERBOSE:
        plt.imshow(mask,cmap='gray')
        plt.title("Region of Interest Mask")
        plt.show()
    return mask

def get_roi_mask_from_bg_images(images_RGB, range=(0, 10), filter=25, dilation=0, log_level=LOG_LEVEL_FATAL):
    mask = np.zeros((images_RGB[0].shape[0],images_RGB[0].shape[1],1),np.uint8)
    for image_RGB in images_RGB:
        mask = cv.bitwise_or(mask,get_roi_mask_from_bg_image(image_RGB,range,filter,log_level=log_level))
    mask = cv.morphologyEx(mask,cv.MORPH_DILATE,get_kernel(3),iterations=dilation)
    if log_level >= LOG_LEVEL_DEBUG:
        plt.imshow(mask,cmap='gray')
        plt.title("Region of Interest Aggregate Mask")
        plt.show()
    return mask

def get_mactching_images_RGB(file_template, log_level=LOG_LEVEL_FATAL):
    files = glob.glob(file_template)
    if not files:
        if log_level >= LOG_LEVEL_WARN:
            print("WARNING: NO FILES MATCH TEMPLATE %s"%file_template)
        return list()
    images_RGB = [cv.cvtColor(cv.imread(f),cv.COLOR_BGR2RGB) for f in files]
    return images_RGB

    

def get_intrinsic_camera_matrix(width, height, focal_length, pixel_size):
    m = np.ones((3,3))
    m[0,0] = focal_length / pixel_size #https://stackoverflow.com/questions/25874196/camera-calibration-intrinsic-matrix-what-do-the-values-represent
    m[1,1] = focal_length / pixel_size
    m[0,2] = width /2 
    m[1,2] = height /2 
    m[2,2] = 1
    return m

def get_intrinsic_camera_matrix_from_image(image, focal_length, pixel_size):
    return get_intrinsic_camera_matrix(image.shape[1], image.shape[0],focal_length,pixel_size)


def normalize_hu_moments(raw_hu_moments, ignore_h7_sign=False):
    if not np.all(raw_hu_moments):
        return np.zeros((len(raw_hu_moments)))
    return np.array([-1 * math.copysign(1.0, raw_hu_moments[i]) * math.log10(abs(raw_hu_moments[i])) for i in np.arange(0,len(raw_hu_moments))])

def score_vector_similarity(v1, v2):
    score = 0
    for i in np.arange(0,min(len(v1),len(v2))):
        score += abs(v1[i]-v2[i])
    return score

#see https://vincent-maladiere.github.io/opencv-tutorial/camera-calibration-and-3d-reconstruction
def calibrate_from_checkerboard(image, point_count):
    # image_thresh = cv.inRange(image,200,255)
    # image_white = np.zeros(image_thresh.shape, dtype=np.uint8)
    # image_white[:] = 255
    # contours, _ = cv.findContours(image_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours = [cv.convexHull(c) for c in contours if cv.contourArea(c) > 1000000]
    # mask = np.zeros(image_white.shape, dtype=np.uint8)
    # cv.drawContours(mask,contours,-1,255,cv.FILLED)
    # mask = cv.morphologyEx(mask, cv.MORPH_ERODE, get_kernel(3), iterations=10)
    # image_white = cv.bitwise_or(cv.bitwise_xor(image_white,mask),cv.bitwise_and(image_thresh,mask))
    # plt.imshow(image_white,cmap='gray',vmin=0, vmax=255)
    # plt.show()
    # board_float = np.float32(image_white)
    _, corners = cv.findChessboardCorners(image, point_count)#cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    objp = np.zeros((1, point_count[0] * point_count[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:point_count[0], 0:point_count[1]].T.reshape(-1, 2)
    _, camera_matrix, distortion_coeffs, rvecs, tvecs = cv.calibrateCamera([objp], [corners_refined], image.shape[::-1], None, None)
    optimal_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, image.shape[::-1], 1)
    map_x, map_y = cv.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, optimal_camera_matrix, image.shape[::-1], 5)
    return map_x, map_y
