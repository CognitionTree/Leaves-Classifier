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
		self.kernel = 'linear'#linear
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

#Neural Network implementation using MLPClassifier()
class NN_Classifier(Classifier):
	def __init__(self):
		super(NN_Classifier, self).__init__()
		self.clf = MLPClassifier()
		
		# default parameters
		self.activation='relu'
		self.alpha=1e-05
		self.batch_size='auto'
		self.beta_1=0.9
		self.beta_2=0.999
		self.early_stopping=False
		self.epsilon=1e-08
		self.hidden_layer_sizes=(130,2)#(5,2)
		self.learning_rate='constant'
		self.learning_rate_init=0.001
		self.max_iter=-1
		self.momentum=0.9
		self.nesterovs_momentum = True
		self.power_t = 0.5
		self.random_state = 1
		self.shuffle = True
		self.solver = 'lbfgs'
		self.tol=0.0001
		self.validation_fraction=0.1
		self.verbose=False
		self.warm_start=False
		
		self.params = dict( activation = self.activation, alpha = self.alpha, \
		batch_size = self.batch_size, beta_1 = self.beta_1, beta_2 = self.beta_2, \
		early_stopping = self.early_stopping, epsilon = self.epsilon, \
		hidden_layer_sizes = self.hidden_layer_sizes, learning_rate = self.learning_rate, \
		learning_rate_init = self.learning_rate_init, max_iter = self.max_iter, \
		momentum = self.momentum, nesterovs_momentum = self.nesterovs_momentum, \
		power_t = self.power_t, random_state = self.random_state, shuffle = self.shuffle, \
		solver = self.solver, tol = self.tol, validation_fraction = self.validation_fraction, \
		verbose = self.verbose, warm_start = self.warm_start)
		
	def train(self):
		feature_vectors_training = self.training_data.get_feature_vectors()
		labels_training = self.training_data.get_labels()
		self.clf.fit(feature_vectors_training, labels_training)
	
	def predict(self):
		feature_vectors_testing = self.testing_data.get_feature_vectors()
		predictions = self.clf.predict(feature_vectors_testing)
		self.testing_data.set_predictions(predictions)
		
	
	def get_parameters(self):
		return self.params
	
	#setters for different parameters. This does not set the parameters on the classifier
	#just the global variables for the class
	def set_hidden_layer_sizes(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes
		
	def set_tol(self, tol):
		self.tol = tol
		
	def set_alpha(self, alpha):
		self.alpha = alpha
	
	#this function applies all changes to the parameters in the classifier
	def set_parameters(self):
		self.clf.set_params(**self.params)
		
		
