import numpy as np
import cv2
from matplotlib import pyplot as plt

cap =cv2.VideoCapture(0)


img = cv2.imread('/home/lie/Desktop/p/capt0004.jpg')

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''
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

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
   '''

















dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
#dst = cv2.GaussianBlur(img, (5, 5), 0)

#bin = cv2.Canny(dst,50,300)
#gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

#hist = cv2.calcHist([gray],[0],None,[256],[0,256])
#plt.hist(gray.ravel(),256,[0,256])
#plt.show()

reses = []
imgs = []
canny = []
cuts = []

#dst = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)
for gray in cv2.split(dst):
	#retval, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	#bin = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
	ret,bin = cv2.threshold(gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	se = np.ones((7,7), dtype='uint8')
	bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, se)

	#kernel = np.ones((5,5),np.uint8)
	#bin = cv2.erode(bin,kernel,iterations = 1)
	#bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
	#bin = cv2.dilate(bin, kernel)
	reses.append(bin)

	imgcopy = img.copy()

	cut = cv2.bitwise_and(imgcopy,imgcopy,mask=bin)
	
	cuts.append(cut)
	#bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))


	#bin = cv2.Canny(cut, 100, 200, apertureSize=5)
	#bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
	#canny.append(binCanny)
	#bin = cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)


	contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = []
	
	for cnt in contours:
		
		cnt = cv2.approxPolyDP(cnt, .05, True)
		area = cv2.contourArea(cnt)
		cnts.append(cnt)
	#cv2.drawContours(imgcopy,cnts,-1,(0,255,0),3)
	#mask = np.zeros(imgcopy.shape[:2], np.uint8)
	#cv2.drawContours(mask, cnt, -1, 255, -1)




	




#ISOLATE CUTS!!!!
for cut in cuts:
	for gray in cv2.split(cut):
		bin = cv2.Canny(cut, 100, 200, apertureSize=5)
		bin = cv2.dilate(bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
		imgcopy = img.copy()
		contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = []
	
		for cnt in contours:
		
			cnt = cv2.approxPolyDP(cnt, 12, True)
			area = cv2.contourArea(cnt)
			cnts.append(cnt)
		cv2.drawContours(imgcopy,cnts,-1,(0,255,0),3)
		imgs.append(imgcopy)


#WORKS!!!!!!!!!!!!!!!!!
letter = cv2.Canny(cuts[2],100,200,apertureSize=3)	

contours, hierarchy = cv2.findContours(letter.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
imgCopy2 = img.copy()
cnts =[]
for cnt in contours:
	cnt = cv2.approxPolyDP(cnt,.05,True)
	if cv2.contourArea(cnt)>5:
		cnts.append(cnt)
cv2.drawContours(imgCopy2,cnts,-1,(0,255,0),3)
i=1
for cut in cuts:
	cv2.imwrite('/home/lie/Desktop/p/cut'+str(i)+'.jpg',cut)
	i+=1

while cv2.waitKey(30):
	cv2.imshow('ORIGI',imgCopy2)
	cv2.imshow('Letter',letter)
	#cv2.imshow('canny',canny[0])
	cv2.imshow('orig1',imgs[0])
	cv2.imshow('orig2',imgs[1])
	cv2.imshow('orig3',imgs[2])
	cv2.imshow('orig4',imgs[3])
	cv2.imshow('orig5',imgs[4])
	cv2.imshow('orig6',imgs[5])
	#cv2.imshow('canny1',cannys[0])
	#cv2.imshow('canny2',cannys[1])
	#cv2.imshow('canny3',cannys[2])
	cv2.imshow('Thresh1', reses[0])
	cv2.imshow('Thresh2',reses[1])
	cv2.imshow('Thresh3',reses[2])
	