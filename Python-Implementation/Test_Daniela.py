from Data import *
from Utils import *
from Classifiers import *
from FeatureExtractors import *


def testing_split_data_function():
	D = Data()
	data_training, data_testing = split_data(D, 0.8)
	print str(data_training) + '\n'
	print str(data_testing)
	
def testing_SVC_classifier():
	svc = SVC_Classifier()
	svc.set_training_data([1,2,4])
	print svc.get_training_data()

def testing_str_data():
	D = Data()
	data = str(D)
	print data

def testing_statistics():
	D = Data()
	S = Statistics(D)
	accuracy = S.get_accuracy()
	print accuracy

def testing_hu_moments_feature_extractor():
	FE = Feature_Extractors()
	image = read_image_grayscale("1.jpg")
	feature_names, hu_moments = FE.hu_momments_extractor(image)
	print feature_names
	print hu_moments

def testing_Data():
	D = read_kaggle_training_table(train_kaggle_table)
	ids = D.get_table_ids()
	l = len(ids)
	
	if D.get_labels() == []:
		print "yes"
	else:
		print "No"
	
	if D.get_table_ids() == []:
		print "yes"
	else:
		print "No"
	
	print l

def testing_showing_image():
	im = read_image_grayscale('1.jpg')
	cv2.imshow('image', im)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
			
def classifying_data_SVC():
	D = read_kaggle_training_table(train_kaggle_table)
	D_training, D_testing = split_data(D, 0.80)
	
	feature_vectors_training = D_training.get_feature_vectors()
	labels_training = D_training.get_labels()
	feature_vectors_testing = D_testing.get_feature_vectors()
	labels_testing = D_testing.get_labels()
	
	clf = SVC_Classifier()
	clf.set_training_data(D_training)
	clf.set_testing_data(D_testing)
	
	clf.train()
	clf.predict()
	
	predictions = D_testing.get_predictions()
	
	for i in range(len(labels_testing)):
		print labels_testing[i], "------------------------", predictions[i]
	S = Statistics(D_testing, label_to_number)
	
	#print "matrix: "
	#print S.get_confusion_matrix()
	#print "accuracy ", S.get_accuracy()
	
	
	#print str(D)
	#print labels
	#print feature_vectors

def classifying_data_NN():
	D = read_kaggle_training_table(train_kaggle_table)
	print "original length ", D.get_length()
	D_training, D_testing = split_data(D, 0.80)
	print "training length ", D_testing.get_length()
	print "testing length ", D_testing.get_length()
	
	feature_vectors_training = D_training.get_feature_vectors()
	labels_training = D_training.get_labels()
	feature_vectors_testing = D_testing.get_feature_vectors()
	labels_testing = D_testing.get_labels()
	
	clf = NN_Classifier()
	clf.set_training_data(D_training)
	clf.set_testing_data(D_testing)
	
	clf.train()
	clf.predict()
	
	S = Statistics(D_testing, label_to_number)


def testing_data_features():
	D = read_kaggle_training_table(train_kaggle_table)
	
	feature_vectors = D.get_feature_vectors()
	labels = D.get_labels()
	table_ids = D.get_table_ids()
	feature_names = D.get_feature_names()
	
	print "Feature names: "
	print feature_names
	print len(feature_names)
	
	print "Feature vectors position 5:"
	print feature_vectors[10]
	print len(feature_vectors[10])
	
	print "label"
	print labels[10]
	print len(labels[10])
	
	print "table_ids"
	print table_ids[10]
	
	
def testing_classifier_hu_moments():
	D = read_all_kaggle_gray_scale_images()
	D_no_labels, D_labels = split_data_by_labels(D)
	
	#images_binary = D.get_images_binary()
	#table_ids = D.get_table_ids()
	#labels = D.get_labels()
	#numeric_labels = D.get_numeric_labels()
	
	
def testing_get_corner_points():
	img = read_image_grayscale("1.jpg")
	corners = get_corner_points(img, 100)

def testing_NN_with_less_features():
	D = read_kaggle_training_table(train_kaggle_table)
	D_training, D_testing = split_data(D, 0.80)
	
	feature_vectors_training = D_training.get_feature_vectors()
	labels_training = D_training.get_labels()
	feature_vectors_testing = D_testing.get_feature_vectors()
	labels_testing = D_testing.get_labels()
	
	new_feature_vectors_training = []
	new_feature_vectors_testing = []
	
	for FV in feature_vectors_training:
		new_feature_vectors_training.append(FV[0:100])
	
	for FT in feature_vectors_testing:
		new_feature_vectors_testing.append(FT[0:100])
	
	D_training.set_feature_vectors(new_feature_vectors_training)
	D_testing.set_feature_vectors(new_feature_vectors_testing)
	
	clf = NN_Classifier()
	clf.set_training_data(D_training)
	clf.set_testing_data(D_testing)
	
	clf.train()
	clf.predict()
	
	#predictions = D_testing.get_predictions()
	
	S = Statistics(D_testing, label_to_number)
	
def testing_NN_hardcoded_data():
	D_training = Data()
	D_testing = Data()
	
	D_training.set_feature_vectors(array([[1,1], [1,2], [5,5], [4,5], [4,10], [15,15], [15,17]]))
	D_testing.set_feature_vectors(array([[2,1], [5,4], [4,5], [15,14]]))
	
	D_training.set_labels(array([1,1,2,2,2, 3,3]))
	D_testing.set_labels(array([1,2,2,3]))
	
	clf = NN_Classifier()
	clf.set_training_data(D_training)
	clf.set_testing_data(D_testing)
	
	clf.train()
	clf.predict()
	
	predictions = D_testing.get_predictions()
	
	print predictions
	
	S = Statistics(D_testing, label_to_number)

def testing_SVM_hardcoded_data():
	D_training = Data()
	D_testing = Data()
	
	D_training.set_feature_vectors(array([[1,1], [1,2], [5,5], [4,5], [4,10], [15,15], [15,17]]))
	D_testing.set_feature_vectors(array([[2,1], [5,4], [4,5], [15,14]]))
	
	D_training.set_labels(array([1,1,2,2,2, 3,3]))
	D_testing.set_labels(array([1,2,2,3]))
	
	clf = SVC_Classifier()
	clf.set_training_data(D_training)
	clf.set_testing_data(D_testing)
	
	clf.train()
	clf.predict()
	
	predictions = D_testing.get_predictions()
	
	print predictions
	
	S = Statistics(D_testing, label_to_number)

def calculate_accuracy():
	labels = [key for key in label_to_number]
	print labels
	print len(labels)
	print len(label_to_number)

def testing_read_kaggle_test_table():
	D = read_kaggle_test_table()
	print str(D)

def testing_Data():
	D = Data()
	print str(D)
#--------------------- Calling functions --------------------------

#testing_split_data_function()
#testing_SVC_classifier()
#testing_str_data()
#testing_statistics()
#testing_hu_moments_feature_extractor()
#testing_Data()
#classifying_data_SVC()
#classifying_data_NN()
#testing_showing_image()
#testing_data_features()
#testing_classifier_hu_moments()
#testing_get_corner_points()
#testing_NN_with_less_features()
#testing_NN_hardcoded_data()
#calculate_accuracy()
#testing_read_kaggle_test_table()
#testing_Data()