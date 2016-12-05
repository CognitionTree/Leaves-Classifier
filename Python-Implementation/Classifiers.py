
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