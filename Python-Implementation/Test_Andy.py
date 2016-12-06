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
	print train_table_data.get_numeric_labels()

def test_read_all_images():
	data = read_all_kaggle_gray_scale_images(kaggle_images_path)
	
	print data.get_table_ids()
	print data.get_images_binary()[0]
	print data.get_labels()

def test_edge_detector():
	#set_printoptions(threshold=nan)
	print get_edge_points(read_image_grayscale(sample_binary_image))

def test_binary_countours():
	#set_printoptions(threshold=nan)
	#display_image(get_binary_image(read_image_grayscale(sample_color_image)))
	display_image(get_binary_image_contours(read_image_grayscale(sample_color_image_3)))

def test_binary():
	#set_printoptions(threshold=nan)
	#display_image(get_binary_image(read_image_grayscale(sample_color_image)))
	display_image(get_binary_image(read_image_grayscale(sample_color_image_3)))

def test_build_excel_table():
	headers = ['Age', 'Money', 'SSN']
	rows = [[15, 100, '1234'], [20, 1000, '5678'], [40, 10000, '8976']]
	file_name = 'text_excel'
	build_excel_file(rows, file_name, headers)

#test_reading_table()
#test_read_all_images()
#test_edge_detector()
#test_binary()
#test_binary_countours()
test_build_excel_table()	
