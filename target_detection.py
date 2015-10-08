import numpy as np
import cv2
import sys, os, errno
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

img = cv2.imread(os.getcwd() + '/test_pictures/capt0001.jpg')
if img is None:
	print("Failed to read image, check path")
	sys.exit(1)

'''gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,10)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)'''

# denoises image using optimized Non-Local Means alg
# --------------
# src, dst, 
# h = controls luminance filtering, bigger removes more; 
# hForColor = above, but for colored pixels;
# templateWindowSize = pixel size of template patch to compute weights
# 					 should be odd (7);
# searchWindowSize = pixel size of window to compute weighted average
# 				   should be odd (21), increases search time linearly
lum = 10
lumColor = lum
dst = cv2.fastNlMeansDenoisingColored(img,None,lum,lumColor,7,21)
'''dst = cv2.GaussianBlur(img, (5, 5), 0)
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)

bin = cv2.Canny(dst,50,300)
gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.hist(gray.ravel(),256,[0,256])
plt.show()'''

reses = []
imgs = []
canny = []
cuts = []

# split takes one array of multiple color channels, 
# returns multiple arrays of one channel each
for gray in cv2.split(dst):	
	'''retval, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	bin = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)'''

	# turns channel into black/white
	# --------------
	# src,
	# threshold = useless with automatic optimal?,
	# maxVal = maximum value to use with binary threshold
	# threshold type = BINARY (sets to maxVal if passes thresh, 0 otherwise) +
	# 				   OTSU (ignores given thresh and finds optimal on its own)
	# --- return ---
	# ret = calculated threshold value
	# bin = output image			
	ret,bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# runs closing morphological transformation to get rid of small points on object
	# --------------
	# src, type, 
	# kernel = convolution matrix to run through image
	# 		   (bigger size == more noise taken out, more computation)
	se = np.ones((7,7), dtype='uint8')
	bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, se)

	'''kernel = np.ones((5,5),np.uint8)
	bin = cv2.erode(bin,kernel,iterations = 1)
	bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
	bin = cv2.dilate(bin, kernel)'''
	reses.append(bin)

	imgcopy = img.copy()
	cut = cv2.bitwise_and(imgcopy,imgcopy,mask=bin)
	cuts.append(cut)

	'''bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
	bin = cv2.Canny(cut, 100, 200, apertureSize=5)
	bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
	canny.append(binCanny)
	bin = cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)'''

	# src;
	# mode = structure of returned array. RETR_LIST is just an array;
	# method = how contours are expressed. APPROX_SIMPLE stores only vertices
	contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = []
	
	for cnt in contours:		
		# uses Douglas-Puecker alg to estimate polygonal shape from vertices
		# --------------
		# contour,
		# epsilon = error for alg, larger error = less smoothing
		# closed = Boolean flag to connect first and last vertex (makes shape closed)
		cnt = cv2.approxPolyDP(cnt, .05, True)
		area = cv2.contourArea(cnt) # area = numerical value
		cnts.append(cnt)

	'''cv2.drawContours(imgcopy,cnts,-1,(0,255,0),3)
	mask = np.zeros(imgcopy.shape[:2], np.uint8)
	cv2.drawContours(mask, cnt, -1, 255, -1)'''

#ISOLATE CUTS!!!!
for cut in cuts:
	for gray in cv2.split(cut):
		
		# Canny edge detection
		# --------------
		# src;
		# minVal = value that definitely is NOT an edge
		# maxVal = value that definitely is an edge
		# apertureSize = size of the Sobel kernel
		bin = cv2.Canny(cut, 100, 200, apertureSize=5)

		# src;
		# kernel = function with two parameters:
		# 			shape = ellipse-ish;
		# 			size = size of kernel 
		bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

		# see lines 98-109, epsilon is higher	
		contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = []
	
		for cnt in contours: 
			cnt = cv2.approxPolyDP(cnt, 12, True)
			area = cv2.contourArea(cnt)
			cnts.append(cnt)

		# draws contours to images and saves them
		# --------------
		# dst image; contours;
		# contourldx = which contour to draw, negative means all contours;
		# color = color of contours in RGB
		# thickness = thickness of line
		imgcopy = img.copy()
		cv2.drawContours(imgcopy,cnts,-1,(0,255,0),3)
		imgs.append(imgcopy)

#WORKS!!!!!!!!!!!!!!!!!
letter = cv2.Canny(cuts[2], 100, 200, apertureSize=3)	
contours, hierarchy = cv2.findContours(letter.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts =[]

for cnt in contours:
	cnt = cv2.approxPolyDP(cnt,.05,True)
	if cv2.contourArea(cnt)>5:
		cnts.append(cnt)

imgCopy2 = img.copy()
cv2.drawContours(imgCopy2,cnts,-1,(0,255,0),3)

# write cuts to file (idk)
try:
    os.makedirs(os.getcwd() + '/ret_pics/')
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise
for i in range(0, len(cuts)):
	cv2.imwrite(os.getcwd() + '/ret_pics/cut'+str(i)+'.jpg',cuts[i])

cv2.imshow('ORIGINAL',imgCopy2)
cv2.moveWindow('ORIGINAL', 50, 0)
cv2.imshow('Letter', letter)
cv2.moveWindow('Letter', 50, len(imgCopy2))
currImg = imgs[0]
for i in range(1, len(imgs)):
	currImg = np.concatenate((currImg, imgs[i]), axis=1)
cv2.imshow('Original Cuts',currImg)
heig = len(imgCopy2) + len(currImg)
cv2.moveWindow('Original Cuts', 50, len(imgCopy2) + len(currImg))
currImg = reses[0]
for i in range(1, len(reses)):
	currImg = np.concatenate((currImg, reses[i]), axis=1)
cv2.imshow('Thresholds',currImg)
cv2.moveWindow('Thresholds', 50, heig + len(currImg))
cv2.waitKey(0)