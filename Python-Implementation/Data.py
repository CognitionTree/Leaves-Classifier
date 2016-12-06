import random

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

	def __init__(self, data):
		self.labels = data.get_labels()
		self.predictions = data.get_predictions()
		self.length = data.get_length()
		self.calculate_confusion_matrix()
		self.calculate_accuracy()
		
	#confusion_matrix is a dictionary with keys = (label, prediction)
	def calculate_confusion_matrix(self):
		self.confusion_matrix = {}
						
		for i in range(len(self.labels)):
			for j in range(len(self.predictions)):
				if (self.labels[i],self.predictions[j]) not in self.confusion_matrix:
					self.confusion_matrix[(self.labels[i],self.predictions[j])] = 1
				else:
					self.confusion_matrix[(self.labels[i],self.predictions[j])] += 1
		
	
	def get_confusion_matrix(self):
		return self.confusion_matrix
	
	#TODO: calculate accuracy in terms of TP, TN, etc once those are found.
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
	
	def get_accuracy(self):
		return self.accuracy
		
		
		


# ---------------------------Useful functions related to Data ---------------------------

# Split Data instance into data_training and data_testing, given a percentage for training
# returns two instances of Data: data_training and data_testing
def split_data(data, percentage_training):
	length_data = data.get_length()
	length_training_data = int(percentage_training * length_data)
	data_training = Data()
	data_testing = Data()
	
	if length_data == 0:
		return data_training, data_testing
	
	#shuffle data:
	shuffled_indices = random.shuffle([i for i in range(length_data)])
	print shuffled_indices
	
	feature_vectors = [data.get_feature_vectors()[i] for i in shuffled_indices]
	labels = [data.get_labels()[i] for i in shuffled_indices]
	predictions = [data.get_predictions()[i] for i in shuffled_indices]
	images_binary = [data.get_images_binary()[i] for i in shuffled_indices]
	images_color = [data.get_images_color()[i] for i in shuffled_indices]
	feature_names = [data.get_feature_names()[i] for i in shuffled_indices]
	table_ids = [data.get_table_ids()[i] for i in shuffled_indices]
	
	
	feature_vectors_training = [feature_vectors[i] for i in range(length_data) if i <= length_training_data]
	feature_vectors_testing = [feature_vectors[i] for i in range(length_data) if i > length_training_data]
	
	labels_training = [labels[i] for i in range(length_data) if i <= length_training_data]
	labels_testing = [labels[i] for i in range(length_data) if i > length_training_data]
	
	predictions_training = [predictions[i] for i in range(length_data) if i <= length_training_data]
	predictions_testing = [predictions[i] for i in range(length_data) if i > length_training_data]
	
	images_binary_training = [images_binary[i] for i in range(length_data) if i <= length_training_data]
	images_binary_testing = [images_binary[i] for i in range(length_data) if i > length_training_data]
	
	images_color_training = [images_color[i] for i in range(length_data) if i <= length_training_data]
	images_color_testing = [images_color[i] for i in range(length_data) if i > length_training_data]
	
	feature_names_training = [feature_names[i] for i in range(length_data) if i <= length_training_data]
	feature_names_testing = [feature_names[i] for i in range(length_data) if i > length_training_data]
	
	table_ids_training = [table_ids[i] for i in range(length_data) if i <= length_training_data]
	table_ids_testing = [table_ids[i] for i in range(length_data) if i > length_training_data]
	
	
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
	
	print "length of data: ", length_data
	print "length of length_training_data: ", length_training_data
	print "length of data_training: ", data_training.get_length()
	print "length of length_training_data: ", (length_data - length_training_data)
	print "length of data_training: ", data_testing.get_length()

	return data_training, data_testing
	
	
	
	
	