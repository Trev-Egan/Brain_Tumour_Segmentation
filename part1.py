import cv2
import numpy as np
import scipy as sp

#Load images

img1 = cv2.imread('Img6.png', cv2.IMREAD_GRAYSCALE)
img1_GT = cv2.imread('Img6_GT.png', cv2.IMREAD_GRAYSCALE )
ret,img1_GT = cv2.threshold(img1_GT,127,255,cv2.THRESH_BINARY)
height, width = img1.shape

#Initialize Variables
dif = 125
t_old = 125
j = 0
i = 0
r1 = np.zeros(img1.shape)
r2 = np.zeros(img1.shape)

#Iterative threshold

while dif > 1:
    for j in range (height):
        for i in range (width):
            if img1[j, i] > t_old:
                r1[j, i] = img1[j, i]
            else:
                r2[j,i] = img1[j, i]
    m1 = (sum(sum(r1)) / max(len(r1), 1))
    m2 = (sum(sum(r2)) / max(len(r2), 1))
    t_new = 0.5 * (m1 + m2)
    dif = abs(t_new - t_old)
    t_old = t_new

#Calculate the Jaccard Matrix:
jaccard_top = 0
jaccard_bottom = 0
for j in range (height):
    for i in range (width):
        if r1[j, i] == img1_GT[j, i]:
            jaccard_top = jaccard_top + 1
        else:
            jaccard_bottom = jaccard_bottom + 1
jaccard = (jaccard_top)/(jaccard_top + jaccard_bottom)
print(jaccard)

# cv2.imshow('image', img1)
# cv2.imshow('Ground Truth', img1_GT)
cv2.imshow('thresholded image', r1)
cv2.waitKey(0)
cv2.destroyAllWindows()
