import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

LOG_LEVEL_SILENT = 0
LOG_LEVEL_FATAL = 1
LOG_LEVEL_ERROR = 2
LOG_LEVEL_WARN = 3
LOG_LEVEL_INFO = 4
LOG_LEVEL_DEBUG = 5
LOG_LEVEL_VERBOSE = 6

def get_kernel(size):
    return np.ones((size,size),np.uint8)

def fill_all_contours(image_BW, log_level=LOG_LEVEL_FATAL):
    contours, _ = cv.findContours(image_BW,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    image_filled = np.zeros(image_BW.shape,dtype=np.uint8)
    cv.drawContours(image_filled,contours,-1,255,cv.FILLED)
    if log_level >= LOG_LEVEL_DEBUG:
        plt.imshow(image_filled, cmap='gray')
        plt.title("Image With All Contours Filled")
        plt.show()
    return image_filled

def get_scores_from_good_keypoints(results, log_level=LOG_LEVEL_FATAL):
    good_match_keypoint_ratio = np.array([r["good_matches"]/min(r["template_keypoints"],r["sample_keypoints"]) for r in results])
    max_ratio = max(good_match_keypoint_ratio)
    if max_ratio == 0:
        scores = np.zeros((len(results),1),dtype=np.float32)
    else:
        scores = np.array([r/max_ratio for r in good_match_keypoint_ratio])
    if log_level >= LOG_LEVEL_DEBUG:
        plt.bar(np.arange(0,len(results)),good_match_keypoint_ratio)
        plt.title("Good Keypoint Matches")
        plt.show()
        plt.bar(np.arange(0,len(results)),scores)
        plt.title("Good Keypoint Match Scores")
        plt.show()
    return scores

def get_scores_from_used_keypoints(results, log_level=LOG_LEVEL_FATAL):
    used_match_keypoint_ratio = np.array([r["used_matches"]/min(r["template_keypoints"],r["sample_keypoints"]) for r in results])
    max_ratio = max(used_match_keypoint_ratio)
    if max_ratio == 0:
        scores = np.zeros((len(results),1),dtype=np.float32)
    else:
        scores = np.array([r/max_ratio for r in used_match_keypoint_ratio])
    if log_level >= LOG_LEVEL_DEBUG:
        plt.bar(np.arange(0,len(results)),used_match_keypoint_ratio)
        plt.title("Used Keypoint Matches")
        plt.show()
        plt.bar(np.arange(0,len(results)),scores)
        plt.title("Use Kepoint Match Scores")
        plt.show()
    return scores

def get_weighted_scores(scores, weights, log_level=LOG_LEVEL_FATAL):
    weight_sum = np.sum(weights)
    weighted_scores = np.zeros(scores[0].shape,dtype=np.float32)
    for i in np.arange(0,len(weights)):
        weighted_scores = np.add(weighted_scores, scores[i]*weights[i]/weight_sum)
    if log_level >= LOG_LEVEL_DEBUG:
        plt.bar(np.arange(0,len(weighted_scores)),weighted_scores)
        plt.title("Total Weighted Face Match Scores")
        plt.show()
    return weighted_scores

