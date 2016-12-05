#This file will be the one that will be runned 'python main.py'
from Utils import *

def test_reading_table():
	table_data = read_kaggle_test_table(test_kaggle_table)
	
	print table_data.get_feature_vectors()[0]
	print len(table_data.get_feature_vectors())
	print table_data.get_feature_names()

test_reading_table()
	
