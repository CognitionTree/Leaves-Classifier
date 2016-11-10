'''
File name: Utils.py
Objective: utility tools for leave classifier
Author: Andy D. Martinez & Daniela Florit
Date created: 11/06/2016
Python Version: 2.7.12
'''

from math import *


#---------------------------------------Math Tools---------------------------------------

#precondition: The 2 vectors have the same dimensions
#This function calculates the dot product of vectors v1 and v2
#v1 is a vector represented as a dictionary. {dimension_1_name: value}
#v2 is a vector represented as a dictionary. {dimension_1_name: value}
#returns a scalar = the value of the dot product
def dot_product(v1, v2):
	dot_product_scalar = 0.0
	
	for dimension in v1:
		dot_product_scalar += v1[dimension] * v2[dimension]
	
	return dot_product_vector
	
#precondition: The 2 vectors have the same dimensions
#This function calculates the sum of vectors v1 and v2
#v1 is a vector represented as a dictionary. {dimension_1_name: value}
#v2 is a vector represented as a dictionary. {dimension_1_name: value}
#returns a vector which is the sum of v1 and v2
def sum_vectors(v1, v2):
	sum_vector = {}
	
	for dimension in v1:
		sum_vector[dimension] = v1[dimension] + v2[dimension]
	
	return sum_vector

#This function multiplies a vector times a constant
#v is a vector
#c is a constant
#returns a new vector = c*v
def scalar_multiplication(v, c):
	scalar_mult_vector = {}
	
	for dimension in v:
		scalar_mult_vector[dimension] = c * v[dimension]
	
	return scalar_mult_vector		

#This function finds the norm of a vector
#v is a vector
#returns the value of the norm of v
def norm(v):
	norm = 0.0
	
	for dimension in v:
		norm = norm+ v[dimension]**2
	
	return sqrt(norm)

#-------------------------------------Feature Extractors---------------------------------

#This function extract relevant classification features from an image
#image is a 2d numpy array
#returns a vector that contains all the features	
def feature_extractor1(image):
	feature_vector = {}
	#Note: Important to have the dimensions hardcoded in variables somewhere accesible
	return feature_vector