# initial version copied from:
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# command line example:
# > python -i test_script.py --image family.jpg

import sys, os
import numpy as np
import argparse
import cv2  # OpenCV
from win32api import GetSystemMetrics
from win32gui import BringWindowToTop

# GetSystemMetrics, see https://docs.microsoft.com/en-us/windows/desktop/api/winuser/nf-winuser-getsystemmetrics
import tkinter
from tkinter import filedialog

import pdb


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = ()
endPt = ()
cropping = False
image_r = None
image = None
roi = None

def click_and_crop(event, x, y, flags, clone_r):
        # grab references to the global variables
        global refPt, endPt, cropping, image_r, image, roi
        
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

                # display the selected roi in the ROI window
                # crop the original image
                nr, nc, n3 = image.shape
                nrr, ncr, n3r = image_r.shape
                refx = int( float(nc)*float(refPt[0])/float(ncr) )
                refy = int( float(nr)*float(refPt[1])/float(nrr) )
                endx = int( float(nc)*float(endPt[0])/float(ncr) )
                endy = int( float(nr)*float(endPt[1])/float(nrr) )
                roi = image[refy:endy, refx:endx]

                roi_display = ResizeToScreen(roi)
                        
                cv2.imshow("ROI", roi_display)
                #hWndROI = cv2.cvGetWindowHandle("ROI")
                #BringWindowToTop(hWndROI)

def ResizeToScreen(image):
        # screen resolution
        screenx = GetSystemMetrics(0)
        screeny = GetSystemMetrics(1)

        # image size
        nr, nc, n3 = image.shape # e.g. (nr, nc, 3)

        #print('image size = (x,y,c)', nnim[1],nnim[0],nnim[2])
        #print('image size = (nr, nc, n3) ', nr, nc, n3)
        #print('screen size = (x,y)', screenx, screeny)

        if np.all( np.array([nr, nc]) <= np.array([screeny, screenx]) ):
                # no need to resize
                image_r = image.copy()

        else:
                rsize = 0.9*min([float(screenx)/float(nc), float(screeny)/float(nr)])
                nr_r, nc_r = int(rsize*nr), int(rsize*nc) # fx, fy
                #print('new image size, (nr, nc) ', nr_r, nc_r)
                
                # Dell laptop is 2048 x 1152
                # home HP is 1366 x 768
                image_r = cv2.resize(image, (nc_r, nr_r)) # nx x ny

        return image_r

def  LoadImage(source_pn, source_fn):
        # load the image, clone it, and setup the mouse callback function
        #image = cv2.imread(args["image"])
        image = cv2.imread(os.path.join(source_pn, source_fn))

        image_r = ResizeToScreen(image)
        
        return image, image_r

############################ Start sript ######################

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="Path to the image")
ap.add_argument("--sourcepath", help="Path to source images")
ap.add_argument("--savepath", help="Path to save images")

args = vars(ap.parse_args()) # args is a dict

# prepare dialogs
# root = tkinter.Tk()
# root.withdraw()

# if sourc path is not specified, use dialog box
if args['sourcepath'] is None:
        root = tkinter.Tk()
        root.withdraw()
        
        file_select = filedialog.askopenfilename()
        source_pn, source_fn = os.path.split(file_select)

else:
        source_pn = args['sourcepath']
        source_fn = None
        
        pass

listIm_fn = [f for f in os.listdir(source_pn) if os.path.isfile(os.path.join(source_pn, f)) and os.path.splitext(f)[1].lower() == '.jpg']
listImBack_fn = list() # for returning to previous images

if not source_fn is None:
        if not source_fn in listIm_fn:
                sys.exit('error: ' + source_fn + ' not in ' + source_pn)
                pass
        pass

else:
        source_fn = listIm_fn[0]

        pass



cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
#hWndImage = cv2.cvGetWindowHandle("image")

# WINDOW_NORMAL scales
# WINDOW_AUTOSIZE (default) displays in original image resolution, even if bigger than screen

image, image_r = LoadImage(source_pn, source_fn)
clone_r = image_r.copy()
cv2.setMouseCallback("image", click_and_crop, clone_r)
cv2.setWindowTitle("image",source_fn)

# keep looping until the 'q' key is pressed
while True:
	# display the image (might have been modified in the callback) and wait for a keypress
	cv2.imshow("image", image_r)
	key = cv2.waitKey(1) & 0xFF # wait 1ms and process events
 
	# # if the 'r' key is pressed, reset the cropping region
	# if key == ord("r"):
	# 	image_r = clone_r.copy()

	# if the 'r' key is pressed, rotate the image
	if key == ord("r"):
                
                
                nr, nc, n3 = image.shape
                # M = cv2.getRotationMatrix2D((nr/2,nc/2),90,1)
                # image = cv2.warpAffine(image, M, (nr, nc)) # , , size of destination image
                image = np.rot90(image)
                image_r = ResizeToScreen(image)
                
                print('rotate image: ', nr, nc, ' to ', image.shape, ' resize to ', image_r.shape)
                
                # nr, nc, n3 = image_r.shape
                # M = cv2.getRotationMatrix2D((nc/2,nr/2),90,1)
                # image_r = cv2.warpAffine(image_r, M, (nr, nc))

                # print('rotate image_r: ', nr, nc, ' to ', image_r.shape)
                
                clone_r = image_r.copy()
                cv2.setMouseCallback("image", click_and_crop, clone_r)

                # bring image window to top
                #BringWindowToTop(hWndImage)
                
	# if the 'c' key is pressed, crop
	elif key == ord("c"):
                # if there are two reference points, then crop the region of interest
                # from teh image and display it
                if refPt and endPt:
                        # crop the original image
                        nr, nc, n3 = image.shape
                        nrr, ncr, n3r = image_r.shape
                        refx = int( float(nc)*float(refPt[0])/float(ncr) )
                        refy = int( float(nr)*float(refPt[1])/float(nrr) )
                        endx = int( float(nc)*float(endPt[0])/float(ncr) )
                        endy = int( float(nr)*float(endPt[1])/float(nrr) )
                        roi = image[refy:endy, refx:endx]

                        roi_display = ResizeToScreen(roi)
                        
                        cv2.imshow("ROI", roi_display)
                        #hWndROI = cv2.cvGetWindowHandle("ROI")
                        #BringWindowToTop(hWndROI)

                        pass
                pass

        # quit
	elif key == ord("q"):
                break

	elif key == ord("s"):
                save_bn, save_ext = os.path.splitext(source_fn)
                save_fn = save_bn + '_crop' + save_ext
                save_pn = source_pn + os.path.sep + 'cropped'
                if not os.path.isdir(save_pn):
                        os.makedirs(save_pn)

                print('writing ', save_pn + os.path.sep + save_fn)
                if roi is None:
                        cv2.imwrite(save_pn + os.path.sep + save_fn, image)
                else:
                        cv2.imwrite(save_pn + os.path.sep + save_fn, roi)

	elif key == ord("n"):
                # find current position in the list
                i_current = listIm_fn.index(source_fn)
                listIm_fn.remove(source_fn)
                listImBack_fn.append(source_fn)

                if i_current > len(listIm_fn)+1:
                        source_fn = listIm_fn[0]
                else:
                        source_fn = listIm_fn[i_current]
                        
                image, image_r = LoadImage(source_pn, source_fn)
                clone_r = image_r.copy()
                cv2.setMouseCallback("image", click_and_crop, clone_r)
                #BringWindowToTop(hWndImage)
                cv2.setWindowTitle("image",source_fn)

                # set roi = None because new image
                roi = None

                pass

	elif key == ord("b"):
                # return to previous image
                # current source_fn is in listIm_fn
                if len(listImBack_fn) == 0:
                        continue
                
                source_fn = listImBack_fn[-1]
                listImBack_fn.remove(source_fn)
                listIm_fn.insert(i_current,source_fn)

                image, image_r = LoadImage(source_pn, source_fn)
                clone_r = image_r.copy()
                cv2.setMouseCallback("image", click_and_crop, clone_r)
                #BringWindowToTop(hWndImage)
                cv2.setWindowTitle("image",source_fn)

                # set roi = None because new image
                roi = None
                
                
# close all open windows
cv2.destroyAllWindows()

