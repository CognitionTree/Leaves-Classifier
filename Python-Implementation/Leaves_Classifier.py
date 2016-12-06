#TODO: Make sure that when the user enters the mode -m lot he is forced to enter a dir
#TODO: We already have the parameters now use them.

import sys

#All available shortcuts for user input
(c, m, d) = ('-c', '-m', '-d')
(svm, nets, logistic, dlearning) = ('svm', 'nets', 'logistic', 'dlearning')
(interactive,lot) = ('interactive','lot')

parameters_names = {'-c':'Classifier', '-m':'Mode', '-d':'Testing Images Directory'}

parameters_values = {'-c':[svm, nets, logistic, dlearning], \
					 '-m':[interactive, lot]}

default_parameters = {'Classifier':'svm', 'Mode':'interactive'}


#It returns a dictionary mapping modes to their values or None if the user input was
#wrong
def read_terminal_parameters():
	parameters = {}
	cur_mode = None
	cur_mode_values = None
	
	for i in range(1, len(sys.argv)):
		#A mode -c, -m, -d ... 
		if i%2 == 1:
			if sys.argv[i] in parameters_names:
				cur_mode = parameters_names[sys.argv[i]]
				cur_mode_values = parameters_values[sys.argv[i]]
			else:
				return None
	
		else:
			if sys.argv[i] in cur_mode_values:
				parameters[cur_mode] = sys.argv[i]
			elif sys.argv[i] == '-d':
				parameters[cur_mode] = sys.argv[i]
			else:
				return None
	
	return parameters

def instantiate_classifier(which_one):
	classifier = None
	
	if which_one == svm:
		#instantiate classifier as svm
		print svm #TODO: delete this
		
	elif which_one == nets:
		#instantiate classifier as nets
		print nets #TODO: delete this
	
	elif which_one == logistic:
		#instantiate classifier as logistic
		print logistic #TODO: delete this
	
	elif which_one == dlearning:
		#instantiate classifier as dlearning
		print dlearning #TODO: delete this
	
	return classifier
		
#classifier is the instance of the classifier to be used
def interactive_mode(classifier): raise NotImplementedError('Implement me')

#classifier is the instance of the classifier to be used
#dir is the directory to find all the test cases
def lot_mode(classifier, dir): raise NotImplementedError('Implement me')

#--------------Main--------------
print sys.argv
parameters = read_terminal_parameters()	
if parameters == None:
	print 'Follow this standards:'
	print 'python Leaves_Classifier.py -c Classifier -m Mode -d Directory'
	print 'Possible Values: ' + str(parameters_values)
	quit()

#TODO: fix the if to deal with all cases in which the number of parameters is not the
#minimun required
if len(parameters) == 0:
	parameters = default_parameters

print 'Using Parameters:'
print parameters

classifier = instantiate_classifier(parameters['Classifier'])

if parameters['Mode'] == interactive:
	interactive_mode(classifier)
	
elif parameters['Mode'] == lot:
	lot_mode(classifier, parameters['Testing Images Directory'])
	
