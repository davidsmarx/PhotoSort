# initial version copied from:
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# command line example:
# > python -i test_script.py --image family.jpg

import sys, os
import numpy as np
import argparse
import cv2  # OpenCV
from win32api import GetSystemMetrics
# GetSystemMetrics, see https://docs.microsoft.com/en-us/windows/desktop/api/winuser/nf-winuser-getsystemmetrics

import pdb


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = ()
endPt = ()
cropping = False
image_r = None

def click_and_crop(event, x, y, flags, clone_r):
        # grab references to the global variables
        global refPt, endPt, cropping, image_r
        
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
                refPt = (x, y)
                #set bounding box by mouse move
                cropping = True
	
        if event == cv2.EVENT_MOUSEMOVE:
                if cropping:
                        mPt = (x, y)
                        image_r = clone_r.copy()
                        cv2.rectangle(image_r, refPt, mPt, (0, 255, 0), 2)

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
	        # the cropping operation is finished
                endPt = (x, y)
                cropping = False
                # draw a rectangle around the region of interest
                image_r = clone_r.copy()
                cv2.rectangle(image_r, refPt, endPt, (0, 255, 0), 2)
                #cv2.imshow("image", image_r)

def ResizeToScreen(nnim, nscr):
        rsize = 0.9*min([float(nscr[1])/float(nnim[1]), float(nscr[0])/float(nnim[0])])
        return (int(rsize*nnim[0]), int(rsize*nnim[1])) # fx, fy


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])

# image size
nnim = image.shape # e.g. (nr, nc, 3)
# screen resolution
screenx = GetSystemMetrics(0)
screeny = GetSystemMetrics(1)

print('image size = (x,y,c)', nnim[1],nnim[0],nnim[2])
print('screen size = (x,y)', screenx, screeny)

# rsize = 0.9*min([float(screenx)/float(nnim[1]), float(screeny)/float(nnim[0])])
# nnimre = (int(rsize*nnim[1]), int(rsize*nnim[0])) # fx, fy
nnimre = ResizeToScreen((nnim[1],nnim[0]), (screenx,screeny)) # fx, fy

# Dell laptop is 2048 x 1152

image_r = cv2.resize(image, nnimre) # nx x ny

clone_r = image_r.copy()

cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
# WINDOW_NORMAL scales
# WINDOW_AUTOSIZE (default) displays in original image resolution, even if bigger than screen

cv2.setMouseCallback("image", click_and_crop, clone_r)

# keep looping until the 'q' key is pressed
while True:
	# display the image (might have been modified in the callback) and wait for a keypress
	cv2.imshow("image", image_r)
	key = cv2.waitKey(1) & 0xFF # wait 1ms and process events
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image_r = clone_r.copy()
 
	# if the 'c' key is pressed, crop
	elif key == ord("c"):
                # if there are two reference points, then crop the region of interest
                # from teh image and display it
                if refPt and endPt:
                        # crop the original image
                        refx = int( float(nnim[1])*float(refPt[0])/float(nnimre[0]) )
                        refy = int( float(nnim[0])*float(refPt[1])/float(nnimre[1]) )
                        endx = int( float(nnim[1])*float(endPt[0])/float(nnimre[0]) )
                        endy = int( float(nnim[0])*float(endPt[1])/float(nnimre[1]) )
                        roi = image[refy:endy, refx:endx]

                        if np.any( np.array(roi.shape) > np.array([screeny, screenx, 3]) ):
                                #if roi > screen size
                                #roi = clone_r[refPt[1]:endPt[1], refPt[0]:endPt[0]]
                                nroi = roi.shape
                                nroire = ResizeToScreen((nroi[1],nroi[0]), (screenx,screeny)) # fx, fy
                                roi_display = cv2.resize(roi, nroire)
                        else:
                                roi_display = roi

                        cv2.imshow("ROI", roi_display)

        # quit
	elif key == ord("q"):
                break

# close all open windows
cv2.destroyAllWindows()

                
