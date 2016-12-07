Team Members:
Daniela Florit
Andy Daniel Martinez

How to Run The Code:
python Leaves_Classifier_final.py mode classification_type classifier feature_extractor

modes
standard = 'st' : This mode will classify the provided Kaggle data
dynamic = 'd' : This mode will classify whatever image is inside Dynamic_Input directory

classification_type
genus = 'g' : classify by genus
species = 's' : classifies by species


classifier
nn = 'nn' : Uses Neural Networks
svm = 'svm' : Uses SVM

feature_extractor:
all_features = 'all' : Uses all features we created
no_moments_features = 'nm' : Uses all features except moments and Hu moments



