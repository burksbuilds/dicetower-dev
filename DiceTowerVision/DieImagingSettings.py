import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from DiceTowerVision.DiceTowerTools import *
from DiceTowerVision.DieFaceTemplate import DieFaceTemplate
from dataclasses import dataclass, field

class DieImagingSettings:
    hue_start:int = field(default=0)
    hue_end:int = field(default=180)
    hue_max:int = field(default=90)
    hue_RGB:tuple[int,int,int] = field(default=(255,255,255))
    number_brightness:int = field(default=255)
    average_circumscribed_radius:int = field(default=0)
    top_number_center_offset:int = field(default=0)



    @staticmethod
    def get_average_circumscribed_radius(samples, log_level=LOG_LEVEL_FATAL):
        if len(samples) == 0:
            return 0.0
        return np.average(np.array([s.circumscribed_radius for s in samples]))
    

    
    __number_test_threshold = 50
    __number_test_capture = 0.05
    @staticmethod

    def get_number_brightness(samples, log_level=LOG_LEVEL_FATAL):
        if len(samples) == 0:
            return int(255)
        brightnesses = np.zeros((256,1),dtype=np.float32)
        for sample in samples:
            sample_mask = sample.get_sample_mask()
            brightnesses = np.add(brightnesses,cv.calcHist([sample.image_hls[:,:,1]],[0], sample_mask,[256],[0,256]))
        brightnesses = brightnesses / np.sum(brightnesses)
        brightnesses_cum = np.cumsum(brightnesses)

        if log_level >= LOG_LEVEL_VERBOSE:
            plt.plot(brightnesses_cum)
            plt.title("Value at test index [%u] = %f (vs %f)"%(DieImagingSettings.__number_test_threshold,brightnesses_cum[DieImagingSettings.__number_test_threshold], DieImagingSettings.__number_test_capture))
            plt.show()
        if brightnesses_cum[DieImagingSettings.__number_test_threshold] > DieImagingSettings.__number_test_capture: #black numbers
            return int(0)
        else:
            return int(255)
        
    
    __hue_brightness_lower_threshold = 50
    __hue_brightness_upper_threshold = 205
    __hue_filter_capture = 0.90
    __hue_brightness_lower_limit = 25
    __hue_brightness_upper_limit = 230
    __brightness_gradient_indicator = 0.0025 #determiens body brightness cutoff
    __high_brightness_lower_limit = 130 #should be 128, but there is smothign form the gradient function (twice). 
    __low_brightness_upper_limit = 125 #should be 127, but there is smothign form the gradient function (twice).
    __backup_brightness_capture = 0.98

    
    @staticmethod
    def get_imaging_settings(samples, log_level=LOG_LEVEL_FATAL):
        settings = DieImagingSettings()
        if len(samples) == 0:
            return int(90)
        settings.average_circumscribed_radius = DieImagingSettings.get_average_circumscribed_radius(samples, log_level=log_level)
        settings.number_brightness = DieImagingSettings.get_number_brightness(samples, log_level=log_level)
        
        #Determine Die Hue
        hues = np.zeros((180,1),dtype=np.float32)
        for sample in samples:
            sample_mask = sample.get_sample_mask()
            body_brightness_mask = cv.inRange(sample.image_hls[:,:,1],
                                              min(DieImagingSettings.__hue_brightness_lower_threshold,255-settings.number_brightness),
                                              max(DieImagingSettings.__hue_brightness_upper_threshold,255-settings.number_brightness))
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
                    plt.title("Sample ROI")
                    plt.subplot(2,3,5)
                    plt.imshow(body_brightness_mask,cmap='gray')
                    plt.title("Number Rejection")
                    plt.subplot(2,3,6)
                    plt.imshow(hue_mask,cmap='gray')
                    plt.title("Hue Detection")
                    plt.show()
        
        hues = hues / np.sum(hues)
        hue_sums = np.copy(hues)
        shift_count=1
        while np.max(hue_sums) < (DieImagingSettings.__hue_filter_capture): 
            hue_sums = np.add(hue_sums,np.roll(hues,-1*shift_count))
            shift_count += 1

        settings.hue_start = int(np.argmax(hue_sums))%180
        settings.hue_end = (settings.hue_start + shift_count)%180
        settings.hue_max = int(np.argmax(hues))
        test_pixel = 255*np.ones((1,1,3), dtype=np.uint8)
        test_pixel[0,0,0] = settings.hue_max
        test_pixel_RGB = cv.cvtColor(test_pixel,cv.COLOR_HSV2RGB).astype(np.float32)/255
        settings.hue_RGB = (test_pixel_RGB[0,0,0],test_pixel_RGB[0,0,1],test_pixel_RGB[0,0,2])
        if log_level >= LOG_LEVEL_DEBUG:
            plt.plot(hues)
            plt.plot([settings.hue_start, settings.hue_end],[hues[settings.hue_start], hues[settings.hue_end]],'or')
            plt.title("Hue Histogram: [%u, %u]"%(settings.hue_start,settings.hue_end))
            plt.show()

        #Determine Brightness Settings
        body_brightnesses = np.zeros((256,1),dtype=np.float32)
        number_brightnesses = np.zeros((256,1),dtype=np.float32)
        
        for sample in samples:
            sample_mask = sample.get_sample_mask()
            hue_mask_raw = get_hue_mask(sample.image_hls[:,:,0], settings.hue_start,settings.hue_end, 10)
            brightness_filtered = cv.GaussianBlur(sample.image_hls[:,:,1],(9,9),0)
            body_brightness_mask = cv.inRange(brightness_filtered,
                                              min(DieImagingSettings.__hue_brightness_lower_limit,255-settings.number_brightness),
                                              max(DieImagingSettings.__hue_brightness_upper_limit,255-settings.number_brightness))
            hue_mask = cv.bitwise_and(hue_mask_raw,body_brightness_mask)
            die_body_mask = cv.morphologyEx(cv.bitwise_and(hue_mask,sample_mask),cv.MORPH_CLOSE,get_kernel(3),iterations=1)
            body_brightnesses = np.add(body_brightnesses,cv.calcHist([sample.image_hls[:,:,1]],[0], die_body_mask,[256],[0,256]))
            die_number_mask = cv.morphologyEx(cv.bitwise_and(cv.bitwise_not(hue_mask),
                                                            cv.morphologyEx(sample_mask,cv.MORPH_ERODE,get_kernel(5),iterations=3)),
                                            cv.MORPH_OPEN,get_kernel(3),iterations=1) #remove noise at the edge of the sample area
            number_brightnesses = np.add(number_brightnesses,cv.calcHist([sample.image_hls[:,:,1]],[0], die_number_mask,[256],[0,256]))

            if log_level >= LOG_LEVEL_VERBOSE:
                plt.subplot(2,4,1)
                plt.imshow(sample.image)
                plt.title("RGBA")
                plt.subplot(2,4,2)
                plt.imshow(sample.image_hls[:,:,0],cmap='gray')
                plt.title("Hue")
                plt.subplot(2,4,3)
                plt.imshow(brightness_filtered,cmap='gray')
                plt.title("Brightness (Filtered)")
                plt.subplot(2,4,4)
                plt.imshow(sample_mask,cmap='gray')
                plt.title("Sample ROI")

                plt.subplot(2,4,5)
                plt.imshow(hue_mask_raw,cmap='gray')
                plt.title("Hue Matching")
                plt.subplot(2,4,6)
                plt.imshow(body_brightness_mask,cmap='gray')
                plt.title("Brightness Guess")
                plt.subplot(2,4,7)
                plt.imshow(die_body_mask,cmap='gray')
                plt.title("Body ID")
                plt.subplot(2,4,8)
                plt.imshow(die_number_mask,cmap='gray')
                plt.title("Number ID")

                plt.gcf().set_size_inches(8,5)
                plt.show()
            

        body_brightnesses = body_brightnesses / np.sum(body_brightnesses)
        body_brightnesses_cum = np.cumsum(body_brightnesses)
        body_brightnesses_cum_grad = np.gradient(body_brightnesses_cum)
        number_brightnesses = number_brightnesses / np.sum(number_brightnesses)
        number_brightnesses_cum = np.cumsum(number_brightnesses)
        number_brightnesses_cum_grad = np.gradient(number_brightnesses_cum)

        # if log_level >= LOG_LEVEL_DEBUG:
        #     f = plt.figure()
        #     f.set_figheight(9)
        #     plt.subplot(3,1,1)
        #     plt.plot(body_brightnesses)
        #     plt.plot(number_brightnesses)
        #     plt.title("Brightness Histogram")
        #     plt.subplot(3,1,2)
        #     plt.plot(body_brightnesses_cum)
        #     plt.plot(number_brightnesses_cum)
        #     plt.title("Cumulative Brightness")
        #     plt.subplot(3,1,3)
        #     plt.plot(body_brightnesses_cum_grad)
        #     plt.plot(number_brightnesses_cum_grad)
        #     plt.title("Cumulative Brightness Gradient")
        #     plt.suptitle("Brightness Threshold = %u"%127)
        #     plt.show()

        # number bruightnesses tend to be clustered on their extreme, but body brightnesses can be on either side (12 and 10 are good examples). gradients near 125-130 are not useful
        if settings.number_brightness < 128: #text is black if at least 50% of the die is very dark
            #brightness_balance = np.abs(body_brightnesses_cum+number_brightnesses_cum-1)
            #balance_threshold = np.argmin(brightness_balance)
            number_exceeds = np.argwhere(number_brightnesses_cum_grad[:DieImagingSettings.__low_brightness_upper_limit] > DieImagingSettings.__brightness_gradient_indicator)
            number_threshold = np.max(number_exceeds) if len(number_exceeds) > 0 else DieImagingSettings.__low_brightness_upper_limit
            body_exceeds = np.concatenate((np.argwhere(body_brightnesses_cum_grad[DieImagingSettings.__high_brightness_lower_limit:] > DieImagingSettings.__brightness_gradient_indicator) + DieImagingSettings.__high_brightness_lower_limit,
                                          np.argwhere(body_brightnesses_cum_grad[number_threshold:DieImagingSettings.__low_brightness_upper_limit] > DieImagingSettings.__brightness_gradient_indicator) + number_threshold))
            body_threshold = np.min(body_exceeds) if len(body_exceeds) > 0 else number_threshold
            # if len(body_exceeds) > 0:
            #     body_threshold = np.min(body_exceeds)+ DieImagingSettings.__high_brightness_lower_limit
            # else: #all of body brightness is below 127
            #     body_exceeds =   np.argwhere(body_brightnesses_cum_grad[number_threshold:DieImagingSettings.__low_brightness_upper_limit] > DieImagingSettings.__brightness_gradient_indicator)
            #     body_threshold = (np.min(body_exceeds) if len(body_exceeds) > 0 else 0 )+ number_threshold
            settings.brightness_threshold = int((body_threshold+number_threshold)/2)
        else:
            #brightness_balance = np.abs(1-body_brightnesses_cum-number_brightnesses_cum)
            #balance_threshold = np.argmin(brightness_balance)
            number_exceeds = np.argwhere(number_brightnesses_cum_grad[DieImagingSettings.__high_brightness_lower_limit:] > DieImagingSettings.__brightness_gradient_indicator)
            number_threshold = (np.min(number_exceeds) if len(number_exceeds) > 0 else 0) + DieImagingSettings.__high_brightness_lower_limit
            body_exceeds = np.concatenate((np.argwhere(body_brightnesses_cum_grad[:DieImagingSettings.__low_brightness_upper_limit] > DieImagingSettings.__brightness_gradient_indicator),
                                           np.argwhere(np.logical_or(body_brightnesses_cum_grad[DieImagingSettings.__high_brightness_lower_limit:number_threshold] > DieImagingSettings.__brightness_gradient_indicator,
                                                                     body_brightnesses_cum[DieImagingSettings.__high_brightness_lower_limit:number_threshold] < DieImagingSettings.__backup_brightness_capture)) 
                                                       + DieImagingSettings.__high_brightness_lower_limit))
            body_threshold = np.max(body_exceeds) if len(body_exceeds) > 0 else number_threshold
            # if len(body_exceeds) > 0:
            #     body_threshold = np.max(body_exceeds) if len(body_exceeds) > 0 else DieImagingSettings.__low_brightness_upper_limit
            # else:
            #     body_threshold = np.max(body_exceeds) if len(body_exceeds) > 0 else DieImagingSettings.__low_brightness_upper_limit
            settings.brightness_threshold = int((body_threshold+number_threshold)/2)
        
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
            plt.ylim((0,10*DieImagingSettings.__brightness_gradient_indicator))
            plt.title("Cumulative Brightness Gradient")
            plt.suptitle("Brightness Threshold = %u"%settings.brightness_threshold)
            plt.show()

        top_number_offsets = np.array([DieImagingSettings.get_top_number_offset(s, settings) for s in samples])
        settings.top_number_center_offset = np.average(top_number_offsets)

        return settings
    
    @staticmethod
    def get_top_number_offset(sample, imaging_settings):
        template = DieFaceTemplate.create_from_sample(sample, imaging_settings)
        return np.linalg.norm(np.array(template.top_face_center))