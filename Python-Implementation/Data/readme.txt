Disclaimer: 
The data used for this project is being taken from both Kaggle's Leaf 
Classification competition, and from the UCI Machine Learning Repository's One-hundred 
plant species leaves data set Data Set. See the wiki section for links to the data sources.These are found in the folders data_binary_Kaggle, and data_binary_UCI_sameKaggle respectively.
Additionally, a second dataset is being used, which contains both the color and binary images from 40 different plant species. This data set was also obtained from UCI’s Machine Learning Repository.

The organization of the Data folder is as follows:

1.Dataset1 folder: 
	1. data_binary_Kaggle:
		This is the main dataset that will be used for the project.
		It includes the binary images labeled with ID numbers 1-1584.
		Also includes a train.csv file, which maps each ID number to the species label, and the corresponding features extracted from the image
		test.csv file is similar to train.csv.


	2. data_binary_UCI_sameKaggle:
		Binary images of plants are organized by species in folders named according to the corresponding species. 
		A readme.txt file is included in the folder, which was provided by the dataset contributors, and explains the organization of the dataset, sources, and useful information about the dataset itself.
		one-hundred_species.pdf includes a sample image for each species included in the dataset.
		data_Mar_64.txt, data_Sha_64.txt, and data_Tex_64, were provided by the dataset contributors and provide the dataset features.


2. Dataset2 folder:
	1.  data_binary/color_UCF_dataset2:
		This is an extra dataset obtained from UCI’s ML repository.
		It includes a folder with the BW folder: Binary images organized in folders named according to each species.
		It includes a folder with the RGB folder: Color images organized in the same way as the Binary images.
		ReadMe.pdf was provided by the dataset contributors and describes the dataset, species, features, etc
		leaf.csv provides a table of features for each image
	