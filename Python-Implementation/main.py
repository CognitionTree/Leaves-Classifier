#This file will be the one that will be runned 'python main.py'
from Utils import *

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
	
test_reading_table()
	
