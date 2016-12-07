import sys

from Data import *
from Classifiers import *
from Utils import *
from FeatureExtractors import *

#Expected Input: python Leaves_Classifier_final.py mode classification_type classifier feature_extractor 

#modes
standard = 'st'
dynamic = 'd'
modes = [standard, dynamic]

#classification_type
genus = 'g'
species = 's'
classification_types = [genus, species]

#classifier
nn = 'nn'
svm = 'svm'
classifiers = [nn, svm]

#feature_extractor
all_features = 'all'
no_moments_features = 'nm'
feature_extractors = [all_features, no_moments_features]

#Variables
dynamic_input_dir = 'Dynamic_Input'
training_tables = 'Data/Training_Tables'
full_table = training_tables + '/full_table.csv'
full_table_genus = training_tables + '/full_table_genus.csv'
partial_table = training_tables + '/partial_table.csv'
partial_table_genus = training_tables + '/partial_table_genus.csv'



#Default Values
default_mode = dynamic
default_classification_type = genus
default_classifier = svm
default_feature_extractor = no_moments_features


def read_user_input():
	mode = None
	classification_type = None
	classifier = None
	feature_extractor = None
	
	#Set default values
	if len(sys.argv) == 1:
		mode = default_mode
		classification_type = default_classification_type
		classifier = default_classifier
		feature_extractor = default_feature_extractor
	
	else:
		mode = sys.argv[1]
		classification_type = sys.argv[2]
		classifier = sys.argv[3]
		feature_extractor = sys.argv[4]
	
	return (mode, classification_type, classifier, feature_extractor)

def get_classifier(which_classifier):
	classifier = None
	
	if which_classifier == nn:
		classifier = NN_Classifier()
	else:
		classifier = SVC_Classifier()
		
	return classifier

def get_test_data(feature_extractor):
	test_data = read_all_grayscale_images(dynamic_input_dir)		
	
	images = test_data.get_images_binary()
	binary_images = []
	feature_vecs = []
	f_e = Feature_Extractors()
	feature_names = None
	#m = 0
	for im in images:
		b_im = get_binary_image_contours(im)
		binary_images.append(b_im)
		#save_image(b_im, str(m)+'.jpg')#delete this
		#m+=1
		if feature_extractor == all_features:
			(feature_names, features) = f_e.all_feature_extractor(b_im)
			feature_vecs.append(features)
		elif feature_extractor == no_moments_features:
			(feature_names, features) = f_e.all_five_feature_extractor(b_im)
			feature_vecs.append(features)	
	
	test_data.set_feature_vectors(array(feature_vecs))
	test_data.set_images_binary(array(binary_images))
	test_data.set_feature_names(array(feature_names))
	
	return test_data
		
			

def get_data(mode, feature_extractor, classification_type):
	data = None
	train_data = None
	test_data = None
	
	if classification_type == genus:
		#do genus
		if feature_extractor == all_features:
			#do all features
			data = read_kaggle_training_table(full_table_genus, genus_to_number)
			
		elif feature_extractor == no_moments_features:
			#do no moments features
			data = read_kaggle_training_table(partial_table_genus, genus_to_number)
			
	elif classification_type == species:
		#do species
		#do genus
		if feature_extractor == all_features:
			#do all features
			data = read_kaggle_training_table(full_table)
			
		elif feature_extractor == no_moments_features:
			#do no moments features
			data = read_kaggle_training_table(partial_table)
	
	if mode == dynamic:
		train_data = data
		test_data = get_test_data(feature_extractor)	
	#standard
	elif mode == standard:
		#do standard
		train_data, test_data = split_data(data, 0.80)
	
	return (train_data, test_data)


def classify(classifier, train_data, test_data):
	
	classifier.set_training_data(train_data)
	classifier.set_testing_data(test_data)
	
	classifier.train()
	classifier.predict()
	
	return test_data


def display_results(prediction_data, mode, classification_type):
	predictions = prediction_data.get_predictions()
	ids = prediction_data.get_table_ids()
	S = None
	
	if mode == dynamic:
		print '------Predictions for images inside ' + dynamic_input_dir + '--------------'
		for i in range(len(ids)):
			print str(ids[i]) + ': ' + str(predictions[i])
	elif mode == standard:
		print '------Overall results:-----------------------------------------------'
		if classification_type == genus:
			S = Statistics(prediction_data, genus_to_number)
			print S.get_statistics()
		elif classification_type == species:
			S = Statistics(prediction_data, label_to_number)
			print S.get_statistics()
		
def display_input_prameters(mode, classification_type, classifier, feature_extractor):
	print '++++++++++++++++++++++++Parameters+++++++++++++++++++++++'
	print 'Mode: ' + mode
	print 'Classification Type: ' + classification_type
	print 'Classifier: ' + classifier
	print 'Feature Extractor: ' + feature_extractor
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
#----------------------------------Main--------------------------------
(mode, classification_type, classifier, feature_extractor) = read_user_input()

display_input_prameters(mode, classification_type, classifier, feature_extractor)

classifier = get_classifier(classifier)

(train_data, test_data) = get_data(mode, feature_extractor, classification_type)

prediction_data = classify(classifier, train_data, test_data)

display_results(prediction_data, mode, classification_type)
#display_image(prediction_data.get_images_binary()[0])
