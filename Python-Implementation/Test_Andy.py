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

def test_solidity_extractor():
	fe = Feature_Extractors()

	print '367:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_367)))
	print '455:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_455)))
	print '-----------------------------------------'
	print '317:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_317)))
	print '1:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_1)))
	print '-----------------------------------------'
	print '2:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_2)))
	print '431:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_431)))
	print '762:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_762)))
	print '-------------------------------------------------------'
	print '5:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_5)))
	print '50:' + str(fe.solidity_feature_extractor(read_image_grayscale(sample_binary_image_50)))

def test_length_width_ratio_feature_extractor():
	fe = Feature_Extractors()

	print '367:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_367)))
	print '455:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_455)))
	print '-----------------------------------------'
	print '317:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_317)))
	print '1:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_1)))
	print '-----------------------------------------'
	print '2:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_2)))
	print '431:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_431)))
	print '762:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_762)))
	print '-------------------------------------------------------'
	print '5:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_5)))
	print '50:' + str(fe.corner_ratio_feature_extractor(read_image_grayscale(sample_binary_image_50)))

def test_moments_feature_extractor():
	fe = Feature_Extractors()

	print '367:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_367)))
	print '455:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_455)))
	print '-----------------------------------------'
	print '317:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_317)))
	print '1:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_1)))
	print '-----------------------------------------'
	print '2:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_2)))
	print '431:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_431)))
	print '762:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_762)))
	print '-------------------------------------------------------'
	print '5:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_5)))
	print '50:' + str(fe.moments_feature_extractor(read_image_grayscale(sample_binary_image_50)))

def test_plot():
	l = ['Rhododendron', 'Zelkova', 'Prunus', 'Magnolia', 'Castanea', 'Liriodendron', 'Phildelphus', 'Morus', 'Crataegus', 'Sorbus', 'Lithocarpus', 'Alnus', 'Populus', 'Arundinaria', 'Ulmus', 'Ginkgo', 'Callicarpa', 'Ilex', 'Betula', 'Eucalyptus', 'Viburnum', 'Cornus', 'Pterocarya', 'Cercis', 'Cotinus', 'Celtis', 'Tilia', 'Olea', 'Fagus', 'Quercus', 'Liquidambar', 'Salix', 'Cytisus', 'Acer']
	ys = []
	xs =[]
	
	for i in range(len(l)):
		ys.append(i)
		xs.append(i)

	plot_points(xs, ys, l)

def ploting_data():
	path_to_tables = 'Data/Plot_Tables'

	files = glob(path_to_tables+'/*.csv')

	for f in files:
		(headers, rows) = read_excel_table(f)
		xs = []
		ys = []
		labels = []
		for row in rows:
			labels.append(row[1])
			xs.append(row[2])
			ys.append(row[3])
		print f
		plot_points(xs, ys, labels)
		

		

	#read_excel_table(table_path)

#test_reading_table()
#test_read_all_images()
#test_edge_detector()
#test_binary()
#test_binary_countours()
#test_build_excel_table()
#test_length_width_ratio_feature_extractor()
#test_solidity_extractor()
#test_moments_feature_extractor()
#test_plot()
ploting_data()	
