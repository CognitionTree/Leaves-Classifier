'''
File name: Utils.py
Objective: utility tools for leave classifier
Author: Andy D. Martinez & Daniela Florit
Date created: 11/06/2016
Python Version: 2.7.12
'''

from math import *
from numpy import *
from Data import *
from glob import *
import cv2
from PIL import Image
from sys import*

#---------------------------------------Variables----------------------------------------
test_kaggle_table = 'Data/Dataset1/data_binary_Kaggle/test.csv'

train_kaggle_table = 'Data/Dataset1/data_binary_Kaggle/train.csv'

kaggle_images_path = 'Data/Dataset1/data_binary_Kaggle' 

sample_binary_image_367 = 'Data/Dataset1/data_binary_Kaggle/367.jpg'
sample_binary_image_455 = 'Data/Dataset1/data_binary_Kaggle/455.jpg'

sample_binary_image_1 = 'Data/Dataset1/data_binary_Kaggle/1.jpg'
sample_binary_image_317 = 'Data/Dataset1/data_binary_Kaggle/317.jpg'

sample_binary_image_2 = 'Data/Dataset1/data_binary_Kaggle/2.jpg'
sample_binary_image_431 = 'Data/Dataset1/data_binary_Kaggle/431.jpg'
sample_binary_image_762 = 'Data/Dataset1/data_binary_Kaggle/762.jpg'

sample_binary_image_5 = 'Data/Dataset1/data_binary_Kaggle/5.jpg'
sample_binary_image_50 = 'Data/Dataset1/data_binary_Kaggle/50.jpg'

sample_color_image_1 = 'Data/Google_Images/1.jpg'
sample_color_image_2 = 'Data/Google_Images/2.jpg'
sample_color_image_3 = 'Data/Google_Images/3.jpg'

label_to_number = {'Populus_Nigra': 69, 'Acer_Saccharinum': 41, 'Quercus_Pontica': 12, \
'Alnus_Viridis': 86, 'Olea_Europaea': 88, 'Acer_Rufinerve': 58, 'Acer_Rubrum': 79, \
'Cotinus_Coggygria': 33, 'Quercus_Castaneifolia': 76, 'Cornus_Macrophylla': 73, \
'Quercus_Pyrenaica': 38, 'Quercus_Rubra': 7, 'Quercus_Semecarpifolia': 49, \
'Quercus_Afares': 13, 'Quercus_Pubescens': 18, 'Acer_Pictum': 57, 'Ginkgo_Biloba': 51, \
'Quercus_Suber': 45, 'Quercus_x_Turneri': 75, 'Salix_Fragilis': 9, 'Alnus_Cordata': 68, \
'Quercus_Agrifolia': 56, 'Sorbus_Aria': 98, 'Acer_Opalus': 0, 'Alnus_Maximowiczii': 55, \
'Tilia_Oliveri': 48, 'Quercus_Trojana': 20, 'Quercus_Phellos': 53, 'Tilia_Tomentosa': 3, \
'Quercus_Greggii': 44, 'Rhododendron_x_Russellianum': 29, 'Quercus_Rhysophylla': 92, \
'Quercus_Crassifolia': 35, 'Alnus_Sieboldiana': 21, 'Castanea_Sativa': 93, \
'Callicarpa_Bodinieri': 39, 'Quercus_Shumardii': 91, 'Zelkova_Serrata': 10, \
'Eucalyptus_Urnigera': 81, 'Liriodendron_Tulipifera': 27, 'Fagus_Sylvatica': 15, \
'Betula_Austrosinensis': 11, 'Crataegus_Monogyna': 74, 'Populus_Adenopoda': 19, \
'Acer_Mono': 72, 'Prunus_Avium': 43, 'Acer_Circinatum': 62, 'Magnolia_Heptapeta': 71, \
'Quercus_Texana': 50, 'Ilex_Aquifolium': 61, 'Lithocarpus_Cleistocarpus': 59, \
'Quercus_Coccifera': 14, 'Quercus_Kewensis': 36, 'Populus_Grandidentata': 78, \
'Cornus_Controversa': 37, 'Quercus_Vulcanica': 85, 'Cytisus_Battandieri': 28, \
'Celtis_Koraiensis': 34, 'Acer_Capillipes': 70, 'Quercus_Dolicholepis': 46, \
'Arundinaria_Simonii': 23, 'Pterocarya_Stenoptera': 1, 'Quercus_Canariensis': 6, \
'Alnus_Rubra': 30, 'Quercus_Cerris': 64, 'Quercus_Ellipsoidalis': 89, \
'Quercus_Palustris': 54, 'Quercus_Ilex': 22, 'Prunus_X_Shmittii': 42, \
'Quercus_Coccinea': 63, 'Quercus_Variabilis': 4, 'Lithocarpus_Edulis': 77, \
'Quercus_x_Hispanica': 90, 'Magnolia_Salicifolia': 5, 'Phildelphus': 16, \
'Acer_Platanoids': 24, 'Tilia_Platyphyllos': 67, 'Acer_Palmatum': 17, \
'Eucalyptus_Glaucescens': 31, 'Ilex_Cornuta': 47, 'Betula_Pendula': 87, \
'Cercis_Siliquastrum': 32, 'Quercus_Phillyraeoides': 25, 'Quercus_Alnifolia': 40, \
'Quercus_Brantii': 8, 'Viburnum_x_Rhytidophylloides': 60, 'Quercus_Chrysolepis': 65, \
'Quercus_Nigra': 95, 'Morus_Nigra': 84, 'Cornus_Chinensis': 26, \
'Ulmus_Bergmanniana': 94,'Liquidambar_Styraciflua': 52, 'Eucalyptus_Neglecta': 66, \
'Quercus_Infectoria_sub': 97, 'Quercus_Hartwissiana': 2, 'Viburnum_Tinus': 83, \
'Quercus_Imbricaria': 80, 'Quercus_Crassipes': 82, 'Salix_Intergra': 96}

#---------------------------------------Math Tools---------------------------------------
#Notice: Since we started using numpy arrays instead of dictionaries to represent vectors
#		 some of these functions are to simple to even be considered. However since they
#		 are already done they will remain on the code.


#precondition: The 2 vectors have the same dimensions
#This function calculates the dot product of vectors v1 and v2. The same index on v1 and
#v2 represents the same dimension
#v1 is a vector represented as numpy array.
#v2 is a vector represented as numpy array.
#returns a scalar = the value of the dot product
def dot_product(v1, v2):
	dot_product_scalar = 1.0*sum(v1*v2)
	return dot_product_scalar
	
#precondition: The 2 vectors have the same dimensions
#This function calculates the sum of vectors v1 and v2
#v1 is a vector represented as a numpy array.
#v2 is a vector represented as a numpy array.
#returns a vector which is the sum of v1 and v2
def sum_vectors(v1, v2):
	sum_vector = v1 + v2
	return sum_vector

#This function multiplies a vector times a constant
#v is a vector
#c is a constant
#returns a new vector = c*v
def scalar_multiplication(v, c):
	scalar_mult_vector = c*v
	return scalar_mult_vector		

#This function finds the norm of a vector
#v is a vector
#returns the value of the norm of v
def norm(v):
	norm = sqrt(sum(v**2))
	return norm

#-------------------------------------Image Processing Tools-----------------------------

def get_cnt(img):
	ret,thresh = cv2.threshold(img,127,255,0)
	contours,hierarchy = cv2.findContours(thresh, 1, 2)

	cnt = contours[0]
	return cnt

def get_moments(img):
	return cv2.moments(get_cnt(img))

def get_solidity(img):
	cnt = get_cnt(img)
	area = cv2.contourArea(cnt)
	hull = cv2.convexHull(cnt)
	hull_area = cv2.contourArea(hull)
	if hull_area == 0:
		hull_area = 0.0001
	solidity = float(area)/hull_area

	return solidity


def max_x_diff(img):
	edge_points = get_edge_points(img)

	min_x = maxint
	max_x = -1

	for point in edge_points:
		x = point[0]
		
		if x > max_x:
			max_x = x
		if x < min_x:
			min_x = x

	x_diff = 1.0*max_x - min_x
	return x_diff

def max_y_diff(img):
	edge_points = get_edge_points(img)

	min_y = maxint
	max_y = -1

	for point in edge_points:
		y = point[1]
		
		if y > max_y:
			max_y = y
		if y < min_y:
			min_y = y

	y_diff = 1.0*max_y - min_y
	return y_diff



def display_image(img):
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

def get_edge_points(img):
	edges = []
	canny_image = cv2.Canny(img, 100, 200)

	for i in range(len(canny_image)):
		for j in range(len(canny_image[0])):
			if canny_image[i][j] == 255:
				edges.append((i,j))
	
	return edges

def get_corner_points(img, maxFeat):
	
	feature_params = dict( maxCorners = maxFeat, qualityLevel = 0.6, minDistance = 7, blockSize = 7 )
	corners = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
	return corners
	
	'''
	for point in corners:
		x, y = point.ravel()
	'''

#make sure that img was read as a grayscale image
def get_binary_image(img):
	img = cv2.medianBlur(img,5)
	#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	

def get_binary_image_contours(image):
	image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)    

	# Perform morphology
	se = ones((7,7), dtype='uint8')
	image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

	# Your code now applied to the closed image
	cnt = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
	mask = zeros(image.shape[:2], uint8)
	cv2.drawContours(mask, cnt, -1, 255, -1)
	
	return mask


def get_image_area(image):
	nonzero = count_nonzero(image)
	return nonzero

#-------------------------------------File System Processing Tools-----------------------

#headers is a list or numpy array
#rows is a list of lists
#file_name does not contain the .csv extension
def build_excel_file(rows, file_name, headers=None):
	formated_string = ''

	if headers != None:
		for h in headers:
			formated_string = formated_string + str(h)+','

		formated_string = formated_string[0:len(formated_string)-1]
		formated_string+='\n'

	for row in rows:
		for e in row:
			formated_string += str(e)+','
		formated_string = formated_string[0:len(formated_string)-1]
		formated_string+='\n'

	f = open(file_name+'.csv', 'w')
	f.write(formated_string)
	f.close()

	
	 

#table_path is the path to the excel_table
#return a Data object with feature vectors and labels included
def read_excel_table(table_path):
	#feature_names = []
	#feature_vectors = []
	#data = Data()
	headers = []
	rows = []
	
	f = open(table_path, 'r')
	
	for line in f:
		line = line.replace('\n', '')
		splited_line = line.split(',')
	
		if len(headers) == 0:
			headers = splited_line
		else:
			#feature_vectors.append([float(feature) for feature in splited_line])
			rows.append(splited_line)
	
	f.close()
	
	#data.set_feature_vectors(array(feature_vectors))
	#data.set_feature_names(array(feature_names))
	
	return (headers, rows)

def read_kaggle_test_table(table_path = test_kaggle_table):
	data = Data()
	feature_vectors = []
	ids = []
	(feature_names, feature_vectors_str) = read_excel_table(table_path)
	feature_names = feature_names[1:len(feature_names)]
	
	for row in feature_vectors_str:
		ids = ids+row[0:1]
		row = row[1:len(row)]
		feature_vectors.append([float(feature) for feature in row])
	
	data.set_feature_vectors(array(feature_vectors))
	data.set_feature_names(array(feature_names))
	data.set_table_ids(array(ids))
	
	return data

#Remember that Column 2 contains the classification of the feature vector
def read_kaggle_training_table(table_path = train_kaggle_table):
	data = Data()
	feature_vectors = []
	labels = []
	ids = []
	numeric_labels = []
	(feature_names, feature_vectors_str) = read_excel_table(table_path)
	feature_names = feature_names[2:len(feature_names)]

	for row in feature_vectors_str:
		labels.append(row[1])
		numeric_labels.append(label_to_number[row[1]])
		ids.append(row[0])
		row = row[2:len(row)]
		
		feature_vectors.append([float(feature) for feature in row])
	
	data.set_feature_vectors(array(feature_vectors))
	data.set_feature_names(array(feature_names))
	data.set_labels(array(labels))
	data.set_table_ids(array(ids))
	data.set_numeric_labels(array(numeric_labels))	

	return data

#image_path is the path to the image
#returns the image as a 2D numpy array (Grayscale)
def read_image_grayscale(image_path):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	return image
	

#returns the image as a 3D numpy array (RGB)
def read_image_color(image_path):
	image = cv2.imread(image_path)
	
	return image

#This function reads all kaggle leaves images in grayscale on the provided path
#It returns a data object that contains images, ids, labels. If no label
#in the Kaggle train table (which means they belong to Kaggle testing set)
#their label will be None
def read_all_kaggle_gray_scale_images(images_directory_path = kaggle_images_path):
	files = glob(images_directory_path+'/*.jpg')
	images = []	
	ids = []
	labels = []
	numeric_labels = []
	data = Data()
		
	
	#in order to get the labels
	train_table_data = read_kaggle_training_table(train_kaggle_table)

	for f in files:
		splited_file_path = f.split('/')
		file_name = splited_file_path[len(splited_file_path)-1]
		splited_file_name = file_name.split('.')

		ids.append(splited_file_name[0])
		images.append(read_image_grayscale(f))
	
		labels.append(None)
		numeric_labels.append(None)

	table_ids = train_table_data.get_table_ids()
	table_labels = train_table_data.get_labels()

	for i in range(len(ids)):
		image_id = ids[i]		

		for j in range(len(table_ids)):
			if table_ids[j] == image_id:
				labels[i] = table_labels[j]
				numeric_labels[i] = label_to_number[table_labels[j]]
				break
		

	data.set_images_binary(array(images))
	data.set_table_ids(array(ids))
	data.set_labels(array(labels))
	data.set_numeric_labels(array(numeric_labels))

	return data

#This function reads all leaves images in rgb on the provided path
#It returns a data object with colored images setted
def read_all_color_images(images_directory_path):
	files = glob(images_directory_path+'/*.jpg')
	images = []	
	data = Data()

	for f in files:
		images.append(read_image_color(f))
	

	data.set_images_color(array(images))
	return data

#This function reads all leaves images in rgb on the provided path
#It returns a data object with colored images setted
def read_all_grayscale_images(images_directory_path):
	files = glob(images_directory_path+'/*.jpg')
	images = []	
	data = Data()

	for f in files:
		images.append(read_image_grayscale(f))
	

	data.set_images_binary(array(images))
	return data
	