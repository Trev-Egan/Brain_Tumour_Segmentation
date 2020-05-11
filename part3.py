import cv2
import numpy as np
import scipy as sp

#Load images

img = cv2.imread('Img6.png', cv2.IMREAD_GRAYSCALE)
img1_GT = cv2.imread('Img6_GT.png', cv2.IMREAD_GRAYSCALE)
ret,img1_GT = cv2.threshold(img1_GT,127,255,cv2.THRESH_BINARY)
height, width = img.shape

pixel_values = img
pixel_values = np.float32(pixel_values) #convert to data type needed for function

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0) ##100 iterations and 0.2 epsilon value

# Calculate centers
k = 6
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)
listOfCordinates = list(zip(centers[0], centers[1]))

# flatten the labels array
labels = labels.flatten()

# reshape back to the original image dimension
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(img.shape)

ret,th = cv2.threshold(segmented_image,127,255,cv2.THRESH_BINARY)

#Calculate Jaccard Scores
jaccard_top = 0
jaccard_bottom = 0
for j in range (height):
    for i in range (width):
        if th[j, i] == img1_GT[j, i]:
            jaccard_top = jaccard_top + 1
        else:
            jaccard_bottom = jaccard_bottom + 1
jaccard = (jaccard_top)/(jaccard_top + jaccard_bottom)
print(jaccard)

# cv2.imshow('original', img)
# cv2.imshow('segmented image', segmented_image)
cv2.imshow('thresholded segment', th)
cv2.imshow('groundtruth',img1_GT)
cv2.waitKey(0)
cv2.destroyAllWindows()
