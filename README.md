# Skin Cancer Detection with 3D-TBP using a CNN (ISIC 2024 Contest)
**Contest link:** https://www.kaggle.com/competitions/isic-2024-challenge

# ► Introduction

This project takes color images of skin lesions and growths and applies a convolutional neural network (CNN) to determine if the lesion is malignant or benign. The images are accompanied by metadata containing the target value (1 = malignant, 0 = benign) and other information. The other information is integrated into the CNN after cleaning and selection of the most relevant features.

# ► The interest of the project

Skin cancer is a frequent form of cancer that can be deadly if not caught early. A model capable of determining if lesions are malignant would be useful in prefiltering patients before consultation with a specialist.

# ► How to Install and Run the Project

## Installation

### *Prerequisites:*
1.	Python 3.11.9 environment
2.	IDE with Jupyter Notebook extension
3.	Conda (for environment creation)
   
### *Files:*
1.	EDA.ipynb : Jupyter notebook containing the exploratory data analysis, data cleaning, feature selection, and preprocessing of metadata
2.	Model.ipynb : Jupyter notebook containing the CNN model. Data augmentation is performed here.
3.	train-image.hdf5 : file containing all the images
4.	train-metadata.csv : metadata file
5.	cleaned-metadata.csv : the cleaned metadata file generated by EDA.ipynb and used by Model.ipynb
6.	hdf5_write.ipynb : code that generates a subset of metadata and images (optional)
7.	requirements.txt : packages to be installed via pip in Conda

## Making the code functional:
After completing the installation, the code will be functional after the following steps:
1.	Place all ipynb files in the same directory
2.	Place all csv and hdf5 files in a subdirectory to store input data ("/data" for example)
3.	Create a new subdirectory for the save files ("/saves" for example)
#### EDA.ipynb:
4.	At top of code in section titled "Create directory paths for the project", update the dataPath variable with the path for directory where the *train-metadata.csv* and *train-image.hdf5* files are stored.
5.	In same section, adjust the file names if necessary
#### Model.ipynb:
6.	In section 2.1, update the dataPath variable with the path for directory where the *cleaned-metadata.csv* and *train-image.hdf5* files are stored.
7.	In same section, update the savePath variable with the path for directory where model outputs will be saved.
8.	In same section, adjust the file names if necessary.
#### hdf5_write.ipynb (optional):
9. At top of code, declare the filepaths and filenames.

## System requirements (minimum)
Operating system: Windows, macOS, Linux
GPU: not required, but much preferred (see Tensorflow for supported GPUs)
CPU: 4-core, modern
RAM: 8GB (16GB allows for accelerated processing)

## Server instances
The model has been tested on AWS SageMaker linked to an S3 bucket, with the input and output files stored in the S3 bucket. The preferred architecture for fast training is:<br>
***????????***
***????????***
***????????***
***????????***

# ► Credit
The primary project was created by:<br>
-	Claire Davat
-	Wilfried Cédric Koffi Djivo
-	Martial Kouassi
-	Andrew Wieber
