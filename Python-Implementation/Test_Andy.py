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
	print get_edge_points(read_image_grayscale(sample_binary_image_1))

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

def test_length_width_ratio_feature_extractor():
	fe = Feature_Extractors()
	'''
	sample_binary_image_367 = 'Data/Dataset1/data_binary_Kaggle/367.jpg'
	sample_binary_image_455 = 'Data/Dataset1/data_binary_Kaggle/455.jpg'

	sample_binary_image_1 = 'Data/Dataset1/data_binary_Kaggle/1.jpg'
	sample_binary_image_317 = 'Data/Dataset1/data_binary_Kaggle/317.jpg'

	sample_binary_image_2 = 'Data/Dataset1/data_binary_Kaggle/2.jpg'
	sample_binary_image_431 = 'Data/Dataset1/data_binary_Kaggle/431.jpg'
	'''


	print '367:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_367)))
	print '455:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_455)))
	print '-----------------------------------------'
	print '317:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_317)))
	print '1:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_1)))
	print '-----------------------------------------'
	print '2:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_2)))
	print '431:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_431)))

#test_reading_table()
#test_read_all_images()
#test_edge_detector()
#test_binary()
#test_binary_countours()
#test_build_excel_table()
test_length_width_ratio_feature_extractor()	
