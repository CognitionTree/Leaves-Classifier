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

#---------------------------------------Variables----------------------------------------
test_kaggle_table = 'Data/Dataset1/data_binary_Kaggle/test.csv'

train_kaggle_table = 'Data/Dataset1/data_binary_Kaggle/train.csv' 


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






#-------------------------------------File System Processing Tools-----------------------

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

def read_kaggle_test_table(table_path):
	data = Data()
	feature_vectors = []
	ids = []
	(feature_names, feature_vectors_str) = read_excel_table(table_path)
	
	for row in feature_vectors_str:
		ids = ids+row[0:1]
		row = row[1:len(row)]
		feature_vectors.append([float(feature) for feature in row])
	
	data.set_feature_vectors(array(feature_vectors))
	data.set_feature_names(array(feature_names))
	data.set_table_ids(array(ids))
	
	return data

#Remember that Column 2 contains the classification of the feature vector
def read_kaggle_training_table(table_path):
	data = Data()
	feature_vectors = []
	labels = []
	ids = []
	(feature_names, feature_vectors_str) = read_excel_table(table_path)
	
	for row in feature_vectors_str:
		labels.append(row[1])
		ids.append(row[0])
		row = row[2:len(row)]
		
		feature_vectors.append([float(feature) for feature in row])
	
	data.set_feature_vectors(array(feature_vectors))
	data.set_feature_names(array(feature_names))
	data.set_labels(array(labels))
	data.set_table_ids(array(ids))
	
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