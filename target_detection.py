import numpy as np
import cv2
import sys, os, errno

# global values for quickly changing values for fine-tuning stuff
default = 7
debugValue = default

cannyLowerEdgeGradient = 100
cannyUpperEdgeGradient = 200

def getLetter(img, cuts):
	#WORKS!!!!!!!!!!!!!!!!!
	letter = cv2.Canny(cuts[2], cannyLowerEdgeGradient, cannyUpperEdgeGradient, apertureSize=3)	
	contours, hierarchy = cv2.findContours(letter.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = []

	for cnt in contours:
		epsilon = .05
		cnt = cv2.approxPolyDP(cnt,epsilon,True)
		if cv2.contourArea(cnt)>5: # get rid of insignificant areas
			cnts.append(cnt)

	imgCopy2 = img.copy()
	green = (0,255,0)
	thickness = 3
	cv2.drawContours(imgCopy2,cnts,-1,green,thickness)
	return letter,imgCopy2

def getEdgesOfCuts(img, cuts):
	imgs = []
	#ISOLATE CUTS!!!!
	for cut in cuts:
		for gray in cv2.split(cut):
			
			# Canny edge detection
			# --------------
			# src;
			# minVal = value of intensity gradient that is NOT an edge
			# maxVal = value of intensity gradient that is an edge
			# apertureSize = size of the Sobel kernel
			bin = cv2.Canny(cut, cannyLowerEdgeGradient, cannyUpperEdgeGradient, apertureSize=5)

			# src;
			# kernel = function with two parameters:
			# 			shape = ellipse-ish;
			# 			size = size of kernel 
			kernelSize = 5
			bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize)))

			# see lines 98-109, epsilon is higher	
			contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			cnts = []
		
			for cnt in contours: 
				epsilon = 12
				cnt = cv2.approxPolyDP(cnt, epsilon, True)
				area = cv2.contourArea(cnt)
				cnts.append(cnt)

			# draws contours to images and saves them
			# --------------
			# dst image; contours;
			# contourldx = which contour to draw, negative means all contours;
			# color = color of contours in RGB
			# thickness = thickness of line
			imgcopy = img.copy()
			green = (0,255,0)
			thickness = 3
			cv2.drawContours(imgcopy,cnts,-1,green,thickness)
			imgs.append(imgcopy)

	return imgs

def getCutsByThreshold(img,dst):
	reses = []
	cuts = []

	# split takes one array of multiple color channels, 
	# returns multiple arrays of one channel each
	for gray in cv2.split(dst):
		cv2.imshow('channel', gray)
		cv2.waitKey(0)
		cv2.destroyWindow('channel')

		# turns channel into black/white
		# --------------
		# src,
		# threshold = useless with automatic optimal?,
		# maxVal = maximum value to use with binary threshold
		# threshold type = BINARY (sets to maxVal if a given pixel passes threshold, 0 otherwise) +
		# 				   OTSU (ignores given thresh, finds optimal threshold for a given bimodal distribution)
		# --- return ---
		# ret = calculated threshold value
		# bin = output image			
		ret,bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# runs closing morphological transformation to get rid of small points in image
		# --------------
		# src, type, 
		# kernel = convolution matrix to run through image
		# 		   (bigger size == more noise taken out, more computation)
		closeKernelSize = 7
		se = np.ones((closeKernelSize,closeKernelSize), dtype='uint8')
		bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, se)

		reses.append(bin)

		imgcopy = img.copy()
		cut = cv2.bitwise_and(imgcopy,imgcopy,mask=bin)
		cuts.append(cut)

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

	return reses,cuts

def displayEverything(dst, letter, imgs, imgCopy2, reses, cuts):
	# write cuts to file (idk)
	try:
	    os.makedirs(os.getcwd() + '/ret_pics/')
	except OSError as exception:
	    if exception.errno != errno.EEXIST:
	        raise
	for i in range(0, len(cuts)):
		cv2.imwrite(os.getcwd() + '/ret_pics/cut'+str(i)+'.jpg',cuts[i])

	# display everything nicely
	imgCopy2 = np.concatenate((imgCopy2, dst), axis=1)
	cv2.imshow('ORIGINAL',imgCopy2)
	cv2.moveWindow('ORIGINAL', 50, 0)
	cv2.waitKey(0)
	cv2.destroyWindow('ORIGINAL')

	cv2.imshow('Letter', letter)
	cv2.moveWindow('Letter', 50, 0)
	cv2.waitKey(0)
	cv2.destroyWindow('Letter')

	currImg = imgs[0]
	for i in range(1, len(imgs)):
		currImg = np.concatenate((currImg, imgs[i]), axis=1)
	cv2.imshow('Original Cuts',currImg)
	cv2.moveWindow('Original Cuts', 50, 0)
	cv2.waitKey(0)
	cv2.destroyWindow('Original Cuts')

	currImg = reses[0]
	for i in range(1, len(reses)):
		currImg = np.concatenate((currImg, reses[i]), axis=1)
	cv2.imshow('Thresholds',currImg)
	cv2.moveWindow('Thresholds', 50, 0)
	cv2.waitKey(0)
	cv2.destroyWindow('Thresholds')

def findAndDisplayLetter(img):

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

	reses,cuts = getCutsByThreshold(img, dst)
	imgs = getEdgesOfCuts(img, cuts)
	letter,imgCopy2 = getLetter(img, cuts)

	displayEverything(dst, letter, imgs, imgCopy2, reses, cuts)

def getNewDebugValFromUser(minVal, maxVal, isInteger=True):
	global debugValue
	debugValue = default
	while True:
		if isInteger:
			newVal = int(raw_input('Enter new value: '))
			if newVal >= minVal and newVal <= maxVal:
				debugValue = newVal
				break
			else:
				print("Out of range. Reenter a value.")
		else: # to do: add floats
			break


def getImageFromUser():
	imgName = raw_input('Enter new picture to test: ')
	img = cv2.imread(os.getcwd() + '/test_pictures/' + imgName)
	if img is None:
		print("Failed to read image, check path")
		sys.exit(1)
	return img

# mostly copied from http://codeplasma.com/2012/12/03/getting-webcam-images-with-python-and-opencv-2-for-real-this-time/
def getImageFromCamera():
	camIndex = 0
	cap = cv2.VideoCapture(camIndex)
	cap.open(camIndex)

	rampFrames = 30
	for i in xrange(rampFrames):
		cap.read()

	success, im = cap.read()
	cap.release()
	if success:
		return im
	else:
		print("Failed to take picture.")

def main():
	while True:
		img = getImageFromCamera()
		findAndDisplayLetter(img)
		if raw_input('Quit? (y/n) ') == 'y':
			break

main()