#This file will be used for testing purposes
from Utils import *
from Data import *
from FeatureExtractors import *
from Classifiers import *

def test_reading_table():
	test_table_data = read_kaggle_test_table(test_kaggle_table)
	
	print test_table_data.get_feature_vectors()[0]
	print len(test_table_data.get_feature_vectors())
	print test_table_data.get_feature_names()
	print test_table_data.get_table_ids()
	
	print '----------------------------------------------------------------------'
	
	train_table_data = read_kaggle_training_table(train_kaggle_table)
	
	print train_table_data.get_feature_vectors()[0]
	print len(train_table_data.get_feature_vectors())
	print train_table_data.get_feature_names()
	print train_table_data.get_labels()
	print train_table_data.get_table_ids()

def test_read_all_images():
	data = read_all_kaggle_gray_scale_images(kaggle_images_path)
	
	print data.get_table_ids()
	print data.get_images_binary()[0]
	print data.get_labels()

#test_reading_table()
test_read_all_images()	
