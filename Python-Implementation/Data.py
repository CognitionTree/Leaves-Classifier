class Data:
	
	
	def __init__(self):
		self.feature_vectors = None
		self.labels = None
		self.predictions = None 
		self.images_binary = None
		self.images_color = None
		self.feature_names = None
	

	#feature_vectors = [phi(I1), phi(I2),...,phi(In)]
	def set_feature_vectors(self, feature_vectors):
		self.feature_vectors = feature_vectors
	
	def get_feature_vectors(self):
		return self.feature_vectors
	
	
	#labels = [l1,l2,...,ln]
	def set_labels(self, labels):
		self.labels = labels
	
	def get_labels(self):
		return self.labels
		
		
	#predictions = [p1, p2, ...., pn]
	def set_predictions(self, predictions):
		self.predictions = predictions	
	
	def get_predictions(self):
		return self.predictions
		
		
	#images_binary = [I1, I2, ..., In]
	#where I is a 2D cv2 array (Binary image)
	def set_images_binary(self, images_binary):
		self.images_binary = images_binary
	
	def get_images_binary(self):
		return self.images_binary
	
	
	#images_color = [I1, I2, ..., In]
	#where I is a 3D cv2 array (RGB image)
	def set_images_color(self, images_color):
		self.images_color = images_color
	
	def get_images_color(self):
		return self.images_color
	
	#feature names = [feature_1_name_str, feature_2_name_str, ...]	
	def set_feature_names(self, feature_names):
		self.feature_names = feature_names
	
	def get_feature_names(self):
		return self.feature_names
		
	
	