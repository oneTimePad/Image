import cv, cv2
import numpy as np
import itertools
import sys
import math
import os.path
import cPickle as pickle
from scipy import ndimage

#ML algorithms
from sklearn import svm
from sklearn import tree
from sklearn import neighbors

from sklearn import preprocessing

shapeSize = 500

def triangle(height=100, width=100, iterations=20, border = 20):
	baseLength = int(math.ceil(width * 0.60))
	minHeight = int(math.ceil(height * 0.10))
	maxHeight = height
	shapes = []
	for y in xrange(minHeight, maxHeight + 1, int(math.ceil((maxHeight - minHeight) / iterations))):
		for x in xrange(0, width + 1, int(math.ceil(width / iterations))):
			#yield np.array([[0,0], [x,y], [baseLength, 0]], np.int32)
			shape = [[0,0], [x,y], [baseLength, 0]]
			shape = [[i[0] + border , i[1] + border] for i in shape]
			shapes.append(np.asarray(shape, np.int32))
	return shapes

def rectangle(height=100, width=100, iterations=20, border=20):
	minLength = int(math.ceil(height * 0.10))
	maxLength = height
	minWidth = int(math.ceil(width * 0.10))
	maxWidth = width
	shapes = []
	for x in xrange(minWidth + 0, maxWidth + 1, int(math.ceil((maxWidth - minWidth)/iterations))):
		for y in xrange(minLength, maxLength + 1, int(math.ceil((maxLength - minLength)/iterations))):
			#yield np.array([[0,0], [x,0], [x,y], [0,y]], np.int32)
			shape = [[0,0], [x,0], [x,y], [0,y]]
			shape = [[i[0] + border , i[1] + border] for i in shape]
			shapes.append(np.array(shape, np.int32))
	return shapes

def rhombus(height=100, width=100, iterations=20, border=20):
	minHeight = int(math.ceil(height * 0.10))
	maxHeight = height
	minTilt = int(math.ceil(height * 0.10))
	maxTilt = int(math.ceil(height * 0.40))
	baseWidth = int(math.ceil(width * 0.60))
	shapes = []
	for x in xrange(minTilt, maxTilt + 1, int(math.ceil((maxTilt - minTilt)/float(iterations)))):
		for y in xrange(minHeight, maxHeight + 1, int(math.ceil((maxHeight - minHeight)/iterations))):
			#yield np.array([[0,0], [baseWidth,0], [x + baseWidth,y], [x,y]], np.int32)
			shape = [[0,0], [baseWidth,0], [x + baseWidth,y], [x,y]]
			shape = [[i[0] + border , i[1] + border] for i in shape]
			shapes.append(np.array(shape, np.int32))
	return shapes

def trapazoid(height=100, width=100, iterations=20, border=20):
	minTop = int(math.ceil(width * 0.10))
	maxTop = int(math.ceil(width * 0.35))
	minHeight = int(math.ceil(height * 0.10))
	maxHeight = height
	baseWidth = width
	shapes = []
	for x in xrange(minTop, maxTop + 1, int(math.ceil((maxTop - minTop)/float(iterations)))):
		for y in xrange(minHeight + 0, maxHeight + 1, int(math.ceil((maxHeight - minHeight)/float(iterations)))):
			#yield np.array([[0,0], [baseWidth,0], [baseWidth - x,y], [x,y]], np.int32)
			shape = [[0,0], [baseWidth,0], [baseWidth - x,y], [x,y]]
			shape = [[i[0] + border , i[1] + border] for i in shape]
			shapes.append(np.array(shape, np.int32))
	return shapes

def boundingArea(rect):
	return abs(rect[0] - rect[2]) * abs(rect[1] * rect[3]);

def getMoments(shape, img_size, border = 20):
	huMnts = [1,2,3,4,5,6]
	img_size +=  border * 2
	img = np.zeros((img_size,img_size), np.uint8)
	cv2.fillConvexPoly(img, shape, 255)
	contours, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		contour = cv2.approxPolyDP(contours[0], 10, True)
		mnts = cv2.moments(contour)
		huMnts = cv2.HuMoments(mnts)

		'''
		#number of sides that the shape has
		h7 = len(contour)
		
		#eccentricity of the shape compared to its bounding box
		boundArea = boundingArea(cv2.boundingRect(contour))
		contArea = cv2.contourArea(contour);
		h8 = (boundArea - contArea) / contArea;

		#print huMnts

		huMnts = huMnts[:6] + [h7, h8]

		print huMnts		
		'''
		huMnts = list(itertools.chain.from_iterable(huMnts))
		huMnts = huMnts[:6]
	return huMnts;

def generate_contours(shapeGen):
	momentList = []
	for shape in shapeGen():
		momentList.append(getMoments(shape))
	return momentList;

def radial_intercepts(shape, img_size, iterations=10, border=20):
	img_size += border*2
	img = np.zeros((img_size,img_size), np.uint8)

	#because opencv contours have a ridiculous structure
	#create look alike contour
	shape_cnt = [np.asarray([[[i[0], i[1]]] for i in shape])]
	#draw contour on image
	cv2.drawContours(img, shape_cnt, -1, 255, 1)
	#enclosing circle and convert parameters to int
	circle = cv2.minEnclosingCircle(shape)
	max_radius = int(circle[1])
	center = tuple(map(int,circle[0]))
	#step to increase circle from zero to min enclosing circle
	radius_step = int(math.ceil(max_radius/iterations))
	intersect_list = []
	#start interating
	for count in range(0, iterations):
		#step
		radius = radius_step * count
		#draw circle
		circle_img = np.zeros((img_size,img_size), np.uint8)
		cv2.circle(circle_img, center, radius, 255)
		#find intersection
		intersect_mask = np.bitwise_and(img,circle_img)
		#indices of intersection points in mask
		intersect_points = np.transpose(intersect_mask.nonzero())
		filtered_points = []
		angles = []
		#convert to tuples
		intersect_points = [(i[1], i[0]) for i in intersect_points]
		for raw_point in intersect_points:
			#check if points are too close together
			same_point = False
			for filter_point in filtered_points:
				if abs(raw_point[0] - filter_point[0][0]) < 5 and abs(raw_point[1] - filter_point[0][1]) < 5:
					same_point = True
					break
			#if they aren't calculate angle radius makes with the x axis		
			if not same_point:
				angle = math.degrees(np.arctan2(raw_point[1] - center[1], (raw_point[0])- center[0]))
				filtered_points.append((raw_point, angle))
				angles.append(angle)

		filtered_points.sort(key=lambda x: x[1])

		#add first point to end, so we can check the angle between the first and last point
		if len(filtered_points) > 0:
			filtered_points.append(filtered_points[0])
			max_angle = 0
			min_angle = 361

			#angles between: (x,y),(w,z) from zip([1,2,3,4,5...,1],[2,3,4,5,...1])->[(1,2),(2,3),(3,4),(4,5)...(last,1)]
			#find min/max angle
			for point_1, point_2 in zip(filtered_points, filtered_points[1:]):
				delta_angle = abs(point_1[1] - point_2[1])
				if delta_angle > max_angle:
					max_angle = delta_angle
				if delta_angle < min_angle:
					min_angle = delta_angle
		else:
			max_angle = 0
			min_angle = 0

		intersect_list.append(len(filtered_points)-1)
		intersect_list.append(min_angle)
		intersect_list.append(max_angle)
	#disp = np.dstack((img,circle_disp,intersect_disp))		
	#cv2.imshow("disp", disp)
	#cv2.waitKey()

	#[number of intersect,min_angle,max_angle]
	return np.asarray(intersect_list, dtype=np.float)


def mod_radial_intercepts(shape, img_size, iterations=10, border=20):
	img_size += border*2
	img = np.zeros((img_size,img_size), np.uint8)

	#because opencv contours have a ridiculous structure
	#create look alike contour
	#shape_cnt = [np.asarray([[[i[0], i[1]]] for i in shape])]
	#draw contour on image
	cv2.drawContours(img, shape, -1, 255, 1)

	cv2.imshow("contour",img)
	#enclosing circle and convert parameters to int
	circle = cv2.minEnclosingCircle(shape)
	max_radius = int(circle[1])
	center = tuple(map(int,circle[0]))
	#step to increase circle from zero to min enclosing circle
	radius_step = int(math.ceil(max_radius/iterations))
	intersect_list = []
	#start interating
	for count in range(0, iterations):
		#step
		radius = radius_step * count
		#draw circle
		circle_img = np.zeros((img_size,img_size), np.uint8)
		cv2.circle(circle_img, center, radius, 255)
		#find intersection
		intersect_mask = np.bitwise_and(img,circle_img)
		#indices of intersection points in mask
		intersect_points = np.transpose(intersect_mask.nonzero())
		filtered_points = []
		angles = []
		#convert to tuples
		intersect_points = [(i[1], i[0]) for i in intersect_points]
		for raw_point in intersect_points:
			#check if points are too close together
			same_point = False
			for filter_point in filtered_points:
				if abs(raw_point[0] - filter_point[0][0]) < 5 and abs(raw_point[1] - filter_point[0][1]) < 5:
					same_point = True
					break
			#if they aren't calculate angle radius makes with the x axis		
			if not same_point:
				angle = math.degrees(np.arctan2(raw_point[1] - center[1], (raw_point[0])- center[0]))
				filtered_points.append((raw_point, angle))
				angles.append(angle)

		filtered_points.sort(key=lambda x: x[1])

		#add first point to end, so we can check the angle between the first and last point
		if len(filtered_points) > 0:
			filtered_points.append(filtered_points[0])
			max_angle = 0
			min_angle = 361

			#angles between: (x,y),(w,z) from zip([1,2,3,4,5...,1],[2,3,4,5,...1])->[(1,2),(2,3),(3,4),(4,5)...(last,1)]
			#find min/max angle
			for point_1, point_2 in zip(filtered_points, filtered_points[1:]):
				delta_angle = abs(point_1[1] - point_2[1])
				if delta_angle > max_angle:
					max_angle = delta_angle
				if delta_angle < min_angle:
					min_angle = delta_angle
		else:
			max_angle = 0
			min_angle = 0

		intersect_list.append(len(filtered_points)-1)
		intersect_list.append(min_angle)
		intersect_list.append(max_angle)
	#disp = np.dstack((img,circle_disp,intersect_disp))		
	#cv2.imshow("disp", disp)
	#cv2.waitKey()

	#[number of intersect,min_angle,max_angle]
	return np.asarray(intersect_list, dtype=np.float)













def disp_intercepts(shape, img_size, iterations=10):
	pass


data = []
classes = []

img_size = 200

for func,name in zip([triangle(width=img_size, height=img_size), rectangle(width=img_size, height=img_size), trapazoid(width=img_size, height=img_size), rhombus(width=img_size, height=img_size)], ['triangle', 'rectangle', 'trapazoid', 'rhombus']):
	print name
	skip = 0
	for shape in func:
		if skip < 2:
			skip += 1
			continue
		skip = 0
		img = np.zeros((img_size + 40, img_size + 40), np.uint8)
		intersect_list = radial_intercepts(shape, img_size)

		moment_list = getMoments(shape, img_size)
		intersect_list = np.concatenate((intersect_list, moment_list))
		data.append(intersect_list)
		classes.append(name)

		#because opencv contours have a ridiculous structure
		shape_cnt = [np.asarray([[[i[0], i[1]]] for i in shape])]
		cv2.drawContours(img, shape_cnt, -1, 255, 1)
		cv2.imshow("img", img)
		print name
		print intersect_list
		intersect_list.reshape(-1,3)
		#print intersect_list
		cv2.waitKey()

data = np.asarray(data)
print [len(i) for i in data]

print len(data)
print len(data[0])
print len(classes)

scaler = preprocessing.StandardScaler()
#scaler = preprocessing.StandardScaler()
scaler.fit(data)
scale_data = scaler.transform(data)

print scale_data

svm = svm.SVC(probability=True)
#knn = neighbors.KNeighborsClassifier(warn_on_equidistant=False)
#dtree = tree.DecisionTreeClassifier()


svm.fit(scale_data, classes)
print svm

svm_errors = []
#knn_errors = []
#dtree_errors = []
#nn_errors = []

lenzip = 0
count = 0
for func,name in zip([triangle(width=img_size, height=img_size), rectangle(width=img_size, height=img_size), trapazoid(width=img_size, height=img_size), rhombus(width=img_size, height=img_size)], ['triangle', 'rectangle', 'trapazoid', 'rhombus']):
	print name
	for shape in func:
		#print "tick"

		#Failed fix this!!!!!!!!!!!!!!!!!!1
		intersect_list = radial_intercepts(shape, img_size)
		intersect_list = np.asarray(scaler.transform(intersect_list))

		svm_prediction = svm.predict(intersect_list)
		#dtree_prediction = dtree.predict(intersect_list)
		if svm_prediction != name:
			prob = svm.predict_proba(intersect_list)
			svm_errors.append([name, svm_prediction, shape, prob, intersect_list])
		else:
			count+=1
'''

img = cv2.imread('/home/lie/Desktop/p/cut1.jpg')

letter = cv2.Canny(img,100,200,apertureSize=3)	

contours, hierarchy = cv2.findContours(letter.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts =[]
for cnt in contours:
	cnt = cv2.approxPolyDP(cnt,.05,True)
	if cv2.contourArea(cnt)>2400:
		cnts.append(cnt)
cv2.drawContours(img,cnts,-1,(0,255,0),3)

print len(cnts)

cnt = cnts[0]
print cnt
intersect_list = mod_radial_intercepts(cnt, 400)
intersect_list = np.asarray(scaler.transform(intersect_list))
moment_list=getMoments(cnt,400)
intersect_list = np.concatenate((intersect_list, moment_list))
svm_prediction = svm.predict(intersect_list)


print svm_prediction

while cv2.waitKey():
	cv2.imshow("img",img)

'''

#print svm_errors
print "SVM: ", len(svm_errors), "/", lenzip
print count
img_size += 20

for err in svm_errors:
	img = np.zeros((img_size,img_size), np.uint8)
	shape = err[2]
	#because opencv contours have a ridiculous structure
	shape_cnt = [np.asarray([[[i[0], i[1]]] for i in shape])]
	cv2.drawContours(img, shape_cnt, -1, 255, 1)
	print err[1], err[3]
	print err[4]
	print ""
	cv2.imshow(str(err[1][0]), img)
	cv2.waitKey()
	cv2.destroyAllWindows()
