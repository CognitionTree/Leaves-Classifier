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

	def hu_momments_extractor(self, image):
		feature_names = array(['hu1', 'hu2','hu3', 'hu4','hu5', 'hu6','hu7'])
		moments = cv2.moments(image)
		hu_moments = cv2.HuMoments(moments).flatten()
		
		return (feature_names, hu_moments)
		
