import sys

from Data import *
from Classifiers import *
from Utils import *
from FeatureExtractors import *

#Expected Input: python Leaves_Classifier_final.py mode classification_type classifier feature_extractor 

#modes
standard = 'standard'
dynamic = 'dynamic'
mode = [standard, dynamic]

#classification_type
genus = 'genus'
species = 'species'
classification_type = [genus, species]
