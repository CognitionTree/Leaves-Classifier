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

	def moments_feature_extractor(self, image):
		moments = get_moments(image)
		
		feature_names = []
		feature_moments = []

		for label in moments:
			feature_names.append(label)
			feature_moments.append(moments[label])		
		

		return (array(feature_names), array(feature_moments))

	def solidity_feature_extractor(self, image):
		solidity = get_solidity(image)
		return (array(['solidity']), array([solidity]))

	def hu_momments_feature_extractor(self, image):
		feature_names = array(['hu1', 'hu2','hu3', 'hu4','hu5', 'hu6','hu7'])
		moments = cv2.moments(image)
		hu_moments = cv2.HuMoments(moments).flatten()
		
		return (feature_names, hu_moments)

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

	def corner_count_feature_extractor(self, image):
		corners = get_corner_points(image, 100)
		features = array([len(corners)])
		feature_names = array(['number_corners'])
		return (feature_names, features)
		
	def perimeter_area_ratio_feature_extractor(self, image):
		
		area_points = get_image_area(image)
		edge_points = get_edge_points(image)
		perimeter = len(edge_points)
		
		perimeter_area_ratio = 1.0 * perimeter / area_points
		
		features = array([perimeter_area_ratio])
		feature_names = array(['perimeter_area_ratio'])

		return (feature_names, features)
	
	def ratio_of_areas_feature_extractor(self, image):
		area_points = get_image_area(image)
		x_diff = max_x_diff(image)
		y_diff = max_y_diff(image)
		
		area_square = x_diff * y_diff
		
		ratio_of_areas = 1.0* area_square / area_points
		
		features = array([ratio_of_areas])
		feature_names = array(['ratio_of_areas'])

		return (feature_names, features)
	
	def corner_ratio_feature_extractor(self, image):
		(feature_names1, features1) = self.corner_count_feature_extractor(image)
		(feature_names2, features2) = self.length_width_ratio_feature_extractor(image)

		return (array(list(feature_names1)+list(feature_names2)), array(list(features1)+list(features2)))
	
	def all_areas_feature_extractor(self, image):
		(feature_names1, features1) = self.perimeter_area_ratio_feature_extractor(image)
		(feature_names2, features2) = self.ratio_of_areas_feature_extractor(image)

		return (array(list(feature_names1)+list(feature_names2)), array(list(features1)+list(features2)))
	
	def all_four_feature_extractor(self, image):
		(feature_names1, features1) = self.corner_count_feature_extractor(image)
		(feature_names2, features2) = self.length_width_ratio_feature_extractor(image)
		(feature_names3, features3) = self.perimeter_area_ratio_feature_extractor(image)
		(feature_names4, features4) = self.ratio_of_areas_feature_extractor(image)

		features = array(list(features1)+list(features2)+list(features3)+list(features4))
		feature_names = array(list(feature_names1)+list(feature_names2)+list(feature_names4)+list(feature_names4))
		
		return (feature_names, features)
	
	def all_five_feature_extractor(self, image):
		(feature_names1, features1) = self.corner_count_feature_extractor(image)
		(feature_names2, features2) = self.length_width_ratio_feature_extractor(image)
		(feature_names3, features3) = self.perimeter_area_ratio_feature_extractor(image)
		(feature_names4, features4) = self.ratio_of_areas_feature_extractor(image)
		(feature_names5, features5) = self.solidity_feature_extractor(image)
		
		features = array(list(features1)+list(features2)+list(features3)+list(features4)+list(features5))
		feature_names = array(list(feature_names1)+list(feature_names2)+list(feature_names4)+list(feature_names4)+list(feature_names5))
		
		return (feature_names, features)
	
	def all_feature_extractor(self, image):
		(feature_names1, features1) = self.corner_count_feature_extractor(image)
		(feature_names2, features2) = self.length_width_ratio_feature_extractor(image)
		(feature_names3, features3) = self.perimeter_area_ratio_feature_extractor(image)
		(feature_names4, features4) = self.ratio_of_areas_feature_extractor(image)
		(feature_names5, features5) = self.solidity_feature_extractor(image)
		(feature_names6, features6) = self.moments_feature_extractor(image)
		(feature_names7, features7) = self.hu_momments_feature_extractor(image)

		features = array(list(features1)+list(features2)+list(features3)+list(features4) + \
		list(features5)+list(features6)+list(features7))
		
		feature_names = array(list(feature_names1)+list(feature_names2)+list(feature_names3) + \
		list(feature_names4) + list(feature_names5) + list(feature_names6) + list(feature_names7))
		
		return (feature_names, features)
