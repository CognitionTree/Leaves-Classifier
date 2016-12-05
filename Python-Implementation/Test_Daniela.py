from Data import *
from Utils import *
from Classifiers import *
from FeatureExtractors import *


def testing_split_data_function():
	D = Data()
	data_training, data_testing = split_data(D, 0.8)
	print data_training, data_testing
	

def testing_SVC_classifier():
	svc = SVC_Classifier()
	svc.set_training_data([1,2,4])
	print svc.get_training_data()




#--------------------- Calling functions --------------------------

#testing_split_data_function()
#testing_SVC_classifier()
