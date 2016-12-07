from random import *

class Data:
	def __init__(self):
		self.length = 0
		self.feature_vectors = []
		self.labels = []
		self.predictions = []
		self.images_binary = []
		self.images_color = []
		self.feature_names = []
		self.table_ids = []
		self.numeric_labels = []
	
	#feature_vectors = [phi(I1), phi(I2),...,phi(In)]
	def set_feature_vectors(self, feature_vectors):
		self.length = len(feature_vectors)
		self.feature_vectors = feature_vectors
	
	def get_feature_vectors(self):
		return self.feature_vectors
	
	
	#labels = [l1,l2,...,ln]
	def set_labels(self, labels):
		self.length = len(labels)
		self.labels = labels
	
	def get_labels(self):
		return self.labels
		
	
	#numeric_labels = [l1,l2,...,ln]
	def set_numeric_labels(self, numeric_labels):
		self.length = len(numeric_labels)
		self.numeric_labels = numeric_labels
	
	def get_numeric_labels(self):
		return self.numeric_labels
		
	#predictions = [p1, p2, ...., pn]
	def set_predictions(self, predictions):
		self.length = len(predictions)
		self.predictions = predictions	
	
	def get_predictions(self):
		return self.predictions
		
		
	#images_binary = [I1, I2, ..., In]
	#where I is a 2D cv2 array (Binary image)
	def set_images_binary(self, images_binary):
		self.length = len(images_binary)
		self.images_binary = images_binary
	
	def get_images_binary(self):
		return self.images_binary

	
	#images_color = [I1, I2, ..., In]
	#where I is a 3D cv2 array (RGB image)
	def set_images_color(self, images_color):
		self.length =len(images_color)
		self.images_color = images_color
	
	def get_images_color(self):
		return self.images_color
	
	#feature names = [feature_1_name_str, feature_2_name_str, ...]	
	def set_feature_names(self, feature_names):
		self.length = len(feature_names)
		self.feature_names = feature_names
	
	def get_feature_names(self):
		return self.feature_names
		
	def set_table_ids(self, table_ids):
		self.length = len(table_ids)
		self.table_ids = table_ids
	
	def get_table_ids(self):
		return self.table_ids

	def get_length(self):
		return self.length
	
	def __str__(self):
		string = ''
		string += 'length: ' + str(self.length) + '\n \n'
		string += 'table_ids: ' + str(self.table_ids) + '\n \n'
		string += 'feature_names: ' + str(self.feature_names) + '\n \n'
		string += 'feature_vectors: ' + str(self.feature_vectors) + '\n \n'
		string += 'labels: ' + str(self.labels) + '\n \n'
		string += 'predictions: ' + str(self.predictions) + '\n \n'
		string += 'images_binary: ' + str(self.images_binary) + '\n \n'
		string += 'images_color: ' + str(self.images_color) + '\n \n'
		
		return string




class Statistics:

	def __init__(self, data, label_to_number):
		self.labels = data.get_labels()
		self.predictions = data.get_predictions()
		self.length = data.get_length()
		self.label_to_number = label_to_number
		self.calculate_confusion_matrix()
		self.calculate_accuracy()
	
	def calculate_confusion_matrix(self):
		self.confusion_matrix = {}
		
		for labels in self.label_to_number:
			for predictions in self.label_to_number:
				self.confusion_matrix[(labels, predictions)] = 0.0
		
		for i in range(len(self.labels)):
			self.confusion_matrix[(self.labels[i], self.predictions[i])] += 1.0
	
	def calculate_accuracy(self):
		total = 0
		self.accuracy = 0
		for (label, prediction) in self.confusion_matrix:
			total += self.confusion_matrix[(label, prediction)]
			if label == prediction:
				self.accuracy += self.confusion_matrix[(label, prediction)]
		
		if total == 0:
			total = 0.00001
		self.accuracy = 1.0* self.accuracy / total
		
	
	def get_statistics(self):
		return {'Accuracy': self.accuracy}

	
# ---------------------------Useful functions related to Data ---------------------------

# Split Data instance into data_training and data_testing, given a percentage for training
# returns two instances of Data: data_training and data_testing
def split_data(data, percentage_training):
	length_data = data.get_length()
	length_training_data = int(percentage_training * length_data)
	feature_names = data.get_feature_names()
	data_training = Data()
	data_testing = Data()
	
	if length_data == 0:
		return data_training, data_testing
	
	#shuffle data:
	shuffled_indices = [i for i in range(length_data)]
	shuffle(shuffled_indices)
	
	#Getting variables and shuffling them
	feature_vectors = [data.get_feature_vectors()[i] for i in shuffled_indices if len(data.get_feature_vectors()) != 0]
	labels = [data.get_labels()[i] for i in shuffled_indices if len(data.get_labels()) != 0]
	predictions = [data.get_predictions()[i] for i in shuffled_indices if len(data.get_predictions()) != 0]
	images_binary = [data.get_images_binary()[i] for i in shuffled_indices if len(data.get_images_binary()) != 0]
	images_color = [data.get_images_color()[i] for i in shuffled_indices if len(data.get_images_color()) != 0]
	table_ids = [data.get_table_ids()[i] for i in shuffled_indices if len(data.get_table_ids()) != 0]
	
	
	#splitting variable content
	feature_vectors_training = [feature_vectors[i] for i in range(length_data) if i <= length_training_data if feature_vectors != []]
	feature_vectors_testing = [feature_vectors[i] for i in range(length_data) if i > length_training_data if feature_vectors != []]
	
	labels_training = [labels[i] for i in range(length_data) if i <= length_training_data if labels != []]
	labels_testing = [labels[i] for i in range(length_data) if i > length_training_data if labels != []]
	
	predictions_training = [predictions[i] for i in range(length_data) if i <= length_training_data if predictions != []]
	predictions_testing = [predictions[i] for i in range(length_data) if i > length_training_data if predictions != []]
	
	images_binary_training = [images_binary[i] for i in range(length_data) if i <= length_training_data if images_binary != []]
	images_binary_testing = [images_binary[i] for i in range(length_data) if i > length_training_data if images_binary != []]
	
	images_color_training = [images_color[i] for i in range(length_data) if i <= length_training_data if images_color != []]
	images_color_testing = [images_color[i] for i in range(length_data) if i > length_training_data if images_color != []]
	
	table_ids_training = [table_ids[i] for i in range(length_data) if i <= length_training_data if table_ids != []]
	table_ids_testing = [table_ids[i] for i in range(length_data) if i > length_training_data if table_ids != []]
	
	feature_names_training = feature_names
	feature_names_testing = feature_names
	
	
	data_training.set_feature_vectors(feature_vectors_training)
	data_training.set_labels(labels_training)
	data_training.set_predictions(predictions_training)
	data_training.set_images_binary(images_binary_training)
	data_training.set_images_color(images_color_training)
	data_training.set_feature_names(feature_names_training)
	data_training.set_table_ids(table_ids_training)
	
	data_testing.set_feature_vectors(feature_vectors_testing)
	data_testing.set_labels(labels_testing)
	data_testing.set_predictions(predictions_testing)
	data_testing.set_images_binary(images_binary_testing)
	data_testing.set_images_color(images_color_testing)
	data_testing.set_feature_names(feature_names_testing)
	data_testing.set_table_ids(table_ids_testing)
	
	#print "length of data: ", length_data
	#print "length of length_training_data: ", length_training_data
	#print "length of data_training: ", data_training.get_length()
	#print "length of length_training_data: ", (length_data - length_training_data)
	#print "length of data_training: ", data_testing.get_length()

	return data_training, data_testing


def split_data_by_labels(data):
	length_data = data.get_length()
	feature_names = data.get_feature_names()
	data_with_labels = Data()
	data_without_labels = Data()
	
	if length_data == 0:
		return data_with_labels, data_without_labels
	
	
	#getting variables
	labels = data.get_labels()
	images_binary = data.get_images_binary()
	table_ids = data.get_table_ids()
	numeric_labels = data.get_numeric_labels()
	
	labels_with = []
	numeric_labels_with = []
	images_binary_with = []
	table_ids_with = []
	
	labels_without = []
	numeric_labels_without = []
	images_binary_without = []
	table_ids_without = []
	
	for i in range(len(labels)):
		if labels[i] == None:
			labels_without.append(labels[i])
			numeric_labels_without.append(numeric_labels[i])
			images_binary_without.append(images_binary[i])
			table_ids_without.append(table_ids[i])
		else:
			labels_with.append(labels[i])
			numeric_labels_with.append(numeric_labels[i])
			images_binary_with.append(images_binary[i])
			table_ids_with.append(table_ids[i])
	
	data_with_labels.set_labels(labels_with)
	data_with_labels.set_numeric_labels(numeric_labels_with)
	data_with_labels.set_table_ids(table_ids_with)
	data_with_labels.set_images_binary(images_binary_with)
	
	
	data_without_labels.set_labels(labels_without)
	data_without_labels.set_numeric_labels(numeric_labels_without)
	data_without_labels.set_table_ids(table_ids_without)
	data_without_labels.set_images_binary(images_binary_without)
	
	return data_with_labels, data_without_labels 
