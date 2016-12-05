import warnings
from Data import Data
from sklearn import *
from sklearn.neural_network import MLPClassifier


#Use: train() predict()	
class Classifier(object):

	def __init__(self):
		self.training_data = None
		self.testing_data = None
	
	def set_training_data(self, training_data):
		self.training_data = training_data
	
	def get_training_data(self):
		return self.training_data
		
	def set_testing_data(self, testing_data):
		self.testing_data = testing_data
	
	def get_testing_data(self):
		return self.testing_data
	
	#No input no output
	def train(self): raise NotImplementedError('Override me')
	
	#returns testing_data with set predictions!!
	def predict(self): raise NotImplementedError('Override me')
	
	#TODO: predict(feature_vector) where feature_vector was not in testing_data


#-------------------------Implement Classifiers-----------------------------------

#SVM implementation using svm.SVC()
class SVC_Classifier(Classifier):
	def __init__(self):
		super(SVC_Classifier, self).__init__()
		self.clf = svm.SVC()
		
		#parameters
		self.kernel = 'rbf'
		self.degree = 3
		self.verbose = False
		self.C = 1.0
		self.probability = False
		self.shrinking = True
		self.max_iter = -1
		self.decision_function_shape = None
		self.random_state = None
		self.tol = 0.001
		self.cache_size = 200
		self.coef0 = 0.0 
		self.gamma = 'auto'
		self.class_weight = None

	def train(self):
		feature_vectors_training = self.training_data.get_feature_vectors()
		labels_training = self.training_data.get_labels()
		self.clf.fit(feature_vectors_training, labels_training)
	
	def predict(self):
		feature_vectors_testing = self.testing_data.get_feature_vectors()
		predictions = self.clf.predict(feature_vectors_testing)
		self.testing_data.set_predictions(predictions)
	
	def get_parameters(self):
		return self.clf.get_params()
	
	#setters for different parameters. This does not set the parameters on the classifier
	#just the global variables for the class
	def set_kernel(self, kernel):
		self.kernel = kernel
		
	def set_degree(self, degree):
		self.degree = degree
		
	def set_random_state(self, random_state):
		self.random_state = random_state
	
	#this function applies all changes to the parameters in the classifier
	def set_parameters(self):
		
		params = dict(C= self.C, kernel = self.kernel, degree = self.degree, \
		gamma = self.gamma, coef0 = self.coef0, shrinking= self.shrinking, \
		probability = self.probability, tol= self.tol, cache_size = self.cache_size, \
		class_weight = self.class_weight, verbose = self.verbose, max_iter= self.max_iter,\
		decision_function_shape = self.decision_function_shape, random_state = self.random_state)
		
		self.clf.set_params(**params)
		
		
