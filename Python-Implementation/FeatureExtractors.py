from Data import *
from Utils import *
import cv2

#TODO: make sure that the names of the feature extractors are self descriptive
'''
General template:
	Feature extractor receives the image as a numpy array
	extracts relevant classification features from image
	returns a tuple containing two numpy arrays:
		the first including the feature names
		the seconf including the features corresponding to those names
'''

class Feature_Extractors:

	def hu_momments_feature_extractor(self, image):
		feature_names = array(['hu1', 'hu2','hu3', 'hu4','hu5', 'hu6','hu7'])
		moments = cv2.moments(image)
		hu_moments = cv2.HuMoments(moments).flatten()
		
		return (feature_names, hu_moments)
	
	
	def corner_count_feature_extractor(self, image):
		corners = get_corner_points(image, 100)
		features = array([len(corners)])
		feature_names = array(['number_corners'])
		return (feature_names, features)

	def length_width_ratio_feature_extractor(self, image):
		ratio = 0
		
		x_diff = max_x_diff(image)
		y_diff = max_y_diff(image)

		
		if y_diff > x_diff:
			ratio = y_diff/x_diff
		else:
			ratio = x_diff/y_diff

		features = array([ratio])
		feature_names = array(['length_width_ratio'])

		return (feature_names, features)

	def corner_ratio_feature_extractor(self, image):
		(feature_names1, features1) = self.corner_count_feature_extractor(image)
		#print 'AAAA'
		#print features1
		#print feature_names1
		(feature_names2, features2) = self.length_width_ratio_feature_extractor(image)

		return (array(list(feature_names1)+list(feature_names2)), array(list(features1)+list(features2)))

		 
			 
