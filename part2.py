import cv2
import numpy as np
import scipy as sp
import random

#Apply thresholding to find seed point(s)
img = cv2.imread('Img1.png', cv2.IMREAD_GRAYSCALE)
# blur = cv2.GaussianBlur(img,(5,5),0)
ret,th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
height, width = img.shape
indexes = []
for j in range (height):
    for i in range (width):
        if th[j, i] == 255:
            coord = (j, i)
            indexes.append(coord)
seed = random.choice(indexes)


def region_growing(img, seed):

    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_size = 1
    intensity_difference = 0
    neighbor_points_list = []

    neighbor_intensity_list = []
    threshold = 127

    height, width = img.shape
    img_size = height * width

    segmented_img = np.zeros((height, width, 1), np.uint8)

    region_mean = img[seed]

#Region growing until intensity difference becomes greater than certain threshold
    while (intensity_difference < region_threshold) & (region_size < img_size):
        #Loop through neighbor pixels
        for i in range(4):
            #Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            #Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

            #Add neighbor if inside and not already in segmented_img
            if check_inside:
                if segmented_img[x_new, y_new] == 0:
                    if segmented_img > threshold:
                        neighbor_points_list.append([x_new, y_new])
                        neighbor_intensity_list.append(img[x_new, y_new])
                        segmented_img[x_new, y_new] = 255
                        region_size += 1


        #Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list-region_mean)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        segmented_img[seed[0], seed[1]] = 255


        #New region mean
        region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)

        #Update the seed value
        seed = neighbor_points_list[index]

        #Remove the value from the neighborhood lists
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]


    return segmented_img


segmented_image = region_growing(img, seed)
#
# cv2.imshow('image', img)
# cv2.imshow('thresholded image', th)
# cv2.imshow('segmented image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
