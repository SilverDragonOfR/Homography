#Program to perform Homography of images by providing one reference image
#Preferably run in Jupyter Notebook
#on clicking Run webcan is switched on
#provide path of Reference image at @1

#To capture press 'c' key on keyboard
#To exit webcam press <spacebar> on keyboard
#Works on MacOs 12

#The output image is stored as "Modified.jpg" in the same folder as .ipynb file

#made by NightFalcon-SilverDragon-PratikTripathy
#OpenCV is great !!

import cv2
import numpy as np
import matplotlib.pyplot as plt

ref_image = cv2.imread("Reference.jpg")# @1
video = cv2.VideoCapture(0)

while True:
    ret,frame = video.read()
    cv2.imshow("Camera Normal",frame)
    
    
    
    k=cv2.waitKey(1)
    if k==32:
        break
    elif k==ord('c'):
        im1_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(ref_image,cv2.COLOR_RGB2GRAY)

        MAX_FEATURES = 500
        orb = cv2.ORB_create(MAX_FEATURES)

        keypoints1, descriptors1 = orb.detectAndCompute(im1_gray,None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2_gray,None)

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1,descriptors2,None)

        matches = sorted(matches, key=lambda x: x.distance)

        numGoodMatches = int(len(matches)*0.1)
        matches = matches[:numGoodMatches]

        points1 = np.zeros((len(matches),2),dtype=np.float32)
        points2 = np.zeros((len(matches),2),dtype=np.float32)

        for i,match in enumerate(matches):
            points1[i:] = keypoints1[match.queryIdx].pt
            points2[i:] = keypoints2[match.trainIdx].pt

        h,mask = cv2.findHomography(points1,points2,cv2.RANSAC)
        height,width,channels = ref_image.shape
        frame_final = cv2.warpPerspective(frame,h,(width,height))
        
        cv2.imwrite("Modified.jpg",frame_final)
        print("clicked!!")
        
    
    
video.release()

cv2.waitKey(1)
cv2.destroyAllWindows()
for i in range(1,5):
    cv2.waitKey(1)
