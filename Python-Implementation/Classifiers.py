class Data:
	#feature_vectors = [phi(I1), phi(I2),...,phi(In)]
	#labels = [l1,l2,...,ln]
	def __init__(self):
		self.feature_vectors = None
		self.labels = None
		self.predictions = None 
		self.images_binaries = None
		self.images_color = None
	
	def set_feature_vectors(self, feature_vector):
		self.feature_vector = feature_vector
		
	
	def get_feature_vectors(self):
		return self.feature_vectors
	
	def get_labels(self):
		return self.labels
	
	
	
	#predictions = [p1, p2, ...., pn]
	def set_predictions(self, predictions):
		self.predictions = predictions
	
	def get_predictions(self):
		return self.predictions
	


#Use: train() predict()	
class Classifier(object):

	def __init__(self, training_data, testing_data):
		self.training_data = training_data
		self.testing_data = testing_data
	
	#No input no output
	def train(self): raise NotImplementedError('Override me')
	
	#returns testing_data with set predictions!!
	def predict(self): raise NotImplementedError('Override me')
	
	#TODO: predict(feature_vector) where feature_vector was not in testing_data

'''
Example for inheriting from Classifier

class Linear_SVM_Classifier(Classifier):
	def __init__(self, features, categories):
		super(Linear_SVM_Classifier, self).__init__(features, categories)
'''

#-------------------------Implement Your Classifiers-----------------------------------