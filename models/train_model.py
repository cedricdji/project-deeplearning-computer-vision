#!/usr/bin/env python
# coding: utf-8

# # MODEL - IMAGE LOADING & NEURAL NETWORK

# In[1]:


#Import libraries
import gc
import csv
import os
import io
import cv2
from PIL import Image
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import boto3
#import tensorrt


# ## 1) GENERAL FUNCTIONS

# In[2]:


#Function to show image
def show_img(image):
    plt.imshow(image, interpolation=None)
    plt.grid(None)
    plt.show()


# In[3]:


#Image cropping
def crop_image(images_list, nbPix = 100):
    output_images = []
    for image in images_list:
        #Height adjustments
        h = len(image)
        adj = len(image) - nbPix
        h1 = round(adj / 2) #Top
        h2 = h - (adj - h1) #Bottom

        #Width adjustments
        w = len(image[0])
        w_adj = w - nbPix
        w1 = round(w_adj / 2) #Left
        w2 = w - (w_adj - w1) #Right

        img = image[h1:h2,w1:w2]
        output_images.append(img)
        
    return np.array(output_images)


# ## 2) IMPORT DATA

# ### 2.1 - Declare file paths

# In[4]:


# URI S3 for the HDF5 file
hdf5_file_s3_uri = "s3://images-projet-deep-learning/train-image.hdf5"


# In[5]:


# URI S3 for the HDF5 file
hdf5_file_s3_uri = "s3://images-projet-deep-learning/train-image.hdf5"


# ### 2.2 - Load metadata from csv

# In[ ]:


#METADATA: color and size features having no NAs
metadata = metadata[["isic_id",
                     "age_approx",
                     "target",
                     "clin_size_long_diam_mm",
                     "tbp_lv_areaMM2",
                     "tbp_lv_area_perim_ratio",
                     "tbp_lv_eccentricity",
                     "tbp_lv_minorAxisMM",
                     "tbp_lv_color_std_mean",
                     "tbp_lv_deltaLBnorm",
                     "tbp_lv_radial_color_std_max",
                     "tbp_lv_location"]]



#Verify that there are no NAs
print("-- X_meta NA counts --")
print(metadata.isna().sum())

#Check number of Unknoxn for tbp_lv_location
loc_unknown=metadata[metadata["tbp_lv_location"]=="Unknown"]
print("Number of unknown for tbp_lv_location", len(loc_unknown))


# In[7]:


#Activate for debugging of the predict function
#metadata["target_cheat"] = metadata["target"]


# ### 2.3 - Clean data

# In[ ]:


metadata=metadata[metadata["tbp_lv_location"]!="Unknown"]

loc_unknown2=metadata[metadata["tbp_lv_location"]=="Unknown"]
print("Number of unknown for tbp_lv_location", len(loc_unknown2))


# In[ ]:


#Apply One-hot encoding for location
location=pd.get_dummies(metadata["tbp_lv_location"],prefix='category')
location = location.astype(int)
metadata = pd.concat([metadata, location], axis=1)
metadata=metadata.drop("tbp_lv_location",axis=1)
print(metadata.columns)


# In[ ]:


# Calculate the mean of age_approx for each target group
mean_age_malign = metadata.loc[metadata["target"] == 1, "age_approx"].mean()
mean_age_benign = metadata.loc[metadata["target"] == 0, "age_approx"].mean()

# Define a function to fill NA based on the target value
def fill_na_by_target(row):
    if pd.isna(row['age_approx']):
        if row['target'] == 1:
            return mean_age_malign
        elif row['target'] == 0:
            return mean_age_benign
    return row['age_approx']

# Apply the function to the age_approx column
metadata['age_approx'] = metadata.apply(fill_na_by_target, axis=1)

#Verify that there are no NAs
print("-- X_meta NA counts --")
print(metadata.isna().sum())


# In[ ]:


'''
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#Normalization
#Select the column
feature=metadata.drop(columns=['isic_id','target'])

#scaler=StandardScaler() for standardization
scaler = MinMaxScaler()
feature_standardized=scaler.fit_transform(feature)
feature_standardized_df = pd.DataFrame(feature_standardized, columns=feature.columns)

metadata=pd.concat([metadata[['isic_id','target']].reset_index(drop=True), feature_standardized_df] , axis=1)
print(len(metadata.columns))
'''


# In[ ]:


#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#Normalization
#Select the column
'''
feature=metadata.drop(columns=['isic_id','target', 'category_Head & Neck',
       'category_Left Arm', 'category_Left Arm - Lower',
       'category_Left Arm - Upper', 'category_Left Leg',
       'category_Left Leg - Lower', 'category_Left Leg - Upper',
       'category_Right Arm', 'category_Right Arm - Lower',
       'category_Right Arm - Upper', 'category_Right Leg',
       'category_Right Leg - Lower', 'category_Right Leg - Upper',
       'category_Torso Back Bottom Third', 'category_Torso Back Middle Third',
       'category_Torso Back Top Third','category_Torso Front',
       'category_Torso Front Bottom Half', 'category_Torso Front Top Half'])
'''
#Select the column
feature=metadata.drop(columns=['isic_id','target', 'category_Head & Neck',
       'category_Left Arm', 'category_Left Arm - Lower',
       'category_Left Arm - Upper', 'category_Left Leg',
       'category_Left Leg - Lower', 'category_Left Leg - Upper',
       'category_Right Arm', 'category_Right Arm - Lower',
       'category_Right Arm - Upper', 'category_Right Leg',
       'category_Right Leg - Lower', 'category_Right Leg - Upper',
       'category_Torso Back Bottom Third', 'category_Torso Back Middle Third',
       'category_Torso Back Top Third',
       'category_Torso Front Bottom Half', 'category_Torso Front Top Half'])
#scaler=StandardScaler() for standardization
scaler = MinMaxScaler()
feature_standardized=scaler.fit_transform(feature)
feature_standardized_df = pd.DataFrame(feature_standardized, columns=feature.columns)

'''
metadata=pd.concat([metadata[['isic_id','target', 'category_Head & Neck',
       'category_Left Arm', 'category_Left Arm - Lower',
       'category_Left Arm - Upper', 'category_Left Leg',
       'category_Left Leg - Lower', 'category_Left Leg - Upper',
       'category_Right Arm', 'category_Right Arm - Lower',
       'category_Right Arm - Upper', 'category_Right Leg',
       'category_Right Leg - Lower', 'category_Right Leg - Upper',
       'category_Torso Back Bottom Third', 'category_Torso Back Middle Third',
       'category_Torso Back Top Third', 'category_Torso Front',
       'category_Torso Front Bottom Half', 'category_Torso Front Top Half']].reset_index(drop=True), feature_standardized_df] , axis=1)
'''

metadata=pd.concat([metadata[['isic_id','target', 'category_Head & Neck',
       'category_Left Arm', 'category_Left Arm - Lower',
       'category_Left Arm - Upper', 'category_Left Leg',
       'category_Left Leg - Lower', 'category_Left Leg - Upper',
       'category_Right Arm', 'category_Right Arm - Lower',
       'category_Right Arm - Upper', 'category_Right Leg',
       'category_Right Leg - Lower', 'category_Right Leg - Upper',
       'category_Torso Back Bottom Third', 'category_Torso Back Middle Third',
       'category_Torso Back Top Third',
       'category_Torso Front Bottom Half', 'category_Torso Front Top Half']].reset_index(drop=True), feature_standardized_df] , axis=1)
print(len(metadata.columns))


# ### 2.4a - Train, Validate, Test Split

# In[13]:


def list_if_needed(obj):
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj

#Function to perform train-validate or train-test-validate split on a list of isic_ids
def ttv_split(isic_ids, test_frac=0.2, validate_frac=0.2, random_state=88, shuffle=True, stratify=None):
    if test_frac < 0 or validate_frac < 0:
        print("ERROR: Test of validate fraction is negative")
        return None
    if test_frac > 1 or validate_frac > 1:
        print("ERROR: Test of validate fraction is above 0")
        return None
    if test_frac + validate_frac >= 1:
        print("ERROR: Test and validate fractions sum to 1 or more.")
        return None

    #Split training from the rest
    test_size = test_frac + validate_frac
    train, temp = train_test_split(isic_ids, test_size = test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
    #Split test and validate
    if test_frac == 0 or validate_frac == 0:
       # return train.tolist(), temp.tolist()
        return list_if_needed(train), list_if_needed(temp)
    else:
        test_size = test_frac / (test_frac + validate_frac)
        test, validate = train_test_split(temp, test_size = test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        #return train.tolist(), test.tolist(), validate.tolist()
        return list_if_needed(train), list_if_needed(test), list_if_needed(validate)

#Generate the splits of the isic_ids
#train_ids, test_ids, val_ids = ttv_split(metadata["isic_id"])


# In[14]:


#keep 10% of isic_Id of target=1 without duplication
isic_id_val_target_1 = metadata[metadata['target'] == 1]['isic_id'].tolist()
n=int(0.1*len(isic_id_val_target_1))
isic_ids_keep_train = np.random.choice(isic_id_val_target_1, n , replace=False)


# Split to isolate the test set
isic_ids = metadata.loc[~metadata["isic_id"].isin(isic_ids_keep_train), 'isic_id']
train_val_ids, test_ids= ttv_split(isic_ids, test_frac=0.2, validate_frac=0.0)


# In[ ]:


print(len(test_ids))


# ### 2.4b - Data augmentation
# - Augment only the malignant data in the training set
# - Reformat all lists (train_ids, test_ids, val_ids) to be compatible: list of tuples

# In[16]:


#Make list of ids compatible with data augmentations
#Base data takes a value of 0, meaning it should not be modified
train_val_ids_mods = [(id, 0) for id in train_val_ids]
test_ids_mods = [(id, 0) for id in test_ids]
#train_ids_mods = [(id, 0) for id in train_ids]
#val_ids_mods = [(id, 0) for id in val_ids]


# In[ ]:


#Identify the malignant cases in the training data
#all_pos = metadata[metadata["target"]==1]["isic_id"]
all_pos = metadata.loc[(metadata["target"] == 1) & (~metadata["isic_id"].isin(isic_ids_keep_train)), 'isic_id']
pos_in_train_val = all_pos[all_pos.isin(train_val_ids)]
print("Number of positives in training data:", len(pos_in_train_val))
#pos_in_train = all_pos[all_pos.isin(train_ids)]
#print("Number of positives in training data:", len(pos_in_train))


# In[ ]:


print(len(metadata))


# In[ ]:


print(len(train_val_ids))


# In[ ]:


print(len(train_val_ids_mods))


# In[ ]:


# Apply augmentations only to training and validation sets before splitting them
#Duplicates of ids will each have a different number, indicating a specific augmentation to be used
nb_of_augments = 50 #apply 500 with the all dataset

rng = np.random.default_rng()
for i in range(nb_of_augments):
    rand_nb = rng.random()
    #Option 1: use random float between 0 and 1
    #train_ids_mods += [(id, rand_nb) for id in pos_in_train_val]
    #Option 2: use integer
    train_val_ids_mods += [(id, i + 1) for id in pos_in_train_val]

#Shuffle the list
np.random.shuffle(train_val_ids_mods)
print(len(train_val_ids_mods))


# In[22]:


train_ids_mods, val_ids_mods= ttv_split(train_val_ids_mods, test_frac=0.0, validate_frac=0.33)


# In[ ]:


print(len(train_ids_mods))


# In[ ]:


print(len(val_ids_mods))


# In[25]:


#apply duplication on reserved validation data (target 1) 

nb_of_augments = 25 #apply 25 with the all dataset

rng = np.random.default_rng()
isic_ids_keep_train_mods=[]
for i in range(nb_of_augments):
    rand_nb = rng.random()
    #Option 1: use random float between 0 and 1
    #train_ids_mods += [(id, rand_nb) for id in pos_in_train_val]
    #Option 2: use integer
    isic_ids_keep_train_mods += [(id, i + 1) for id in isic_ids_keep_train]



#val_size = int(0.5 * len(val_ids_mods))
#random_sample = random.sample(val_ids_mods, val_size)

#Shuffle the list
val_ids_mods=isic_ids_keep_train_mods + val_ids_mods
np.random.shuffle(val_ids_mods)


# In[ ]:


print(len(val_ids_mods))


# ### 2.5 - Load images and create hybrid tensorflow dataset

# In[27]:


def hair_removal(image, crop_pixels=10):
    height_pixels = len(image)  # Image rows
    width_pixels = len(image[0])  # Image columns

    # Image cropping
    height = [crop_pixels, height_pixels - crop_pixels]
    width = [crop_pixels, width_pixels - crop_pixels]
    img = image[height[0]:height[1], width[0]:width[1]]

    # Gray scale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Black hat filter
    kernel = cv2.getStructuringElement(1, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    # Binary thresholding (MASK)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    # Replace pixels of the mask
    dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)

    return dst

#def resize_image(image, target_size=(100, 100)):
#    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
#    return resized_image


# In[28]:


# Define the augmentation function
def augment_image(image):
    """
    Apply a series of augmentations to create diverse variations of the input image.
    Includes random flips, rotations, brightness adjustments, and other transformations.
    """
    # Apply various augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


# In[29]:


def compute_class_weights(metadata, img_names):
    # Initialize counters for target=0 and target=1
    target_0_count = 0
    target_1_count = 0

    # Loop through each tuple in img_names (img_name, transformation)
    for img_name, mod in img_names:
        # Filter metadata to find the corresponding isic_id
        metadata_filtered = metadata[metadata["isic_id"] == img_name]

        if not metadata_filtered.empty:
            # Retrieve the target value for the corresponding isic_id
            target = metadata_filtered["target"].values[0]

            # Increment the counter based on the target value
            if target == 0:
                target_0_count += 1
            elif target == 1:
                target_1_count += 1

    # Calculate total number of images
    total = target_0_count + target_1_count

    # Calculate class weights based on the counts, avoid division by zero
    if target_0_count > 0:
        weight_for_0 = total / (2 * target_0_count)
    else:
        weight_for_0 = 1

    if target_1_count > 0:
        weight_for_1 = total / (2 * target_1_count)
    else:
        weight_for_1 = 1

    return weight_for_0, weight_for_1


# In[30]:


import s3fs
import h5py

# Define S3 URI for the HDF5 file
hdf5_file_s3_uri = "s3://images-projet-deep-learning/train-image.hdf5"


# ## 3) CNN MODEL

# ### 3.1 - Model class

# In[31]:


#Simple CNN model using only images and target
class CNN_model(tf.keras.Model):
    def __init__(self, neurons = 8, activ = 'leaky_relu', img_size = 100, img_channels=3):
        #Run the constructor of the parent class
        super().__init__()

        #Weight and bias initializers
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        
        #Image size declaration
        self.img_size = img_size
        self.img_channels = img_channels

        #Layers
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1), activation='relu', padding='same', input_shape=(img_size, img_size, img_channels),
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(neurons, activation = activ, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    def call(self, inputs):
        x_image, x_meta = inputs

        # Convolutions
        x1 = self.conv1(x_image)
        x1 = self.pool1(x1)

        # Flattening of images for input layer
        x1 = self.flatten(x1)

        # Hidden layers of neural network
        x1 = self.dense1(x1)

        # Output layer of neural network
        output = self.dense2(x1)

        return output

#Metadata Neural Network
class Meta_model(tf.keras.Model):
    def __init__(self, neurons = 8, activ = 'tanh'):
        #Run the constructor of the parent class
        super().__init__()

        #Weight and bias initializers
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

        #Layers
        self.dense1 = tf.keras.layers.Dense(neurons, activation = activ, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.dense2 = tf.keras.layers.Dense(neurons, activation = activ, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.dropout = tf.keras.layers.Dropout(0.25)

    def call(self, inputs, training=False):
        x_image, x_meta = inputs
        x_all = tf.reshape(x_meta, (tf.shape(x_meta)[0], x_meta.shape[-1]))
        # Neural Network
        x_all = self.dense1(x_all)
        x_all = self.dense2(x_all)
        if training:
            x_all = self.dropout(x_all, training=training)
        output = self.dense3(x_all)
        return output

#Hybrid CNN model taking metadata
class Hybrid_model(tf.keras.Model):
    def __init__(self, neurons = 8, activ = 'leaky_relu', img_size = 100, img_channels = 3):
        #Run the constructor of the parent class
        super().__init__()

        #Weight and bias initializers
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

        #Image size declaration
        self.img_size = img_size
        self.img_channels = img_channels

        #Layers
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), activation='relu', padding='same', input_shape=(img_size, img_size, img_channels),
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.conv2 = tf.keras.layers.Conv2D(64, 5, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(neurons, activation = activ, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.dropout1 = tf.keras.layers.Dropout(0.10)
        self.dense2 = tf.keras.layers.Dense(neurons, activation = activ, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.dropout2 = tf.keras.layers.Dropout(0.10)
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.concatenate = keras.layers.Concatenate(axis=1)
        
    def call(self, inputs, training=False):
        x_image, x_meta = inputs
        # Convolutions
        x = self.conv1(x_image)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        # Flattening of images and concatenation with other data
        x = self.flatten(x)
        # Reshape metadata to match dimensions
        x_meta = tf.reshape(x_meta, (tf.shape(x_meta)[0], x_meta.shape[-1]))
        x_all = self.concatenate([x, x_meta])
        # Neural Network
        x_all = self.dense1(x_all)
        if training:
            x_all = self.dropout1(x_all, training=training)
        x_all = self.dense2(x_all)
        if training:
            x_all = self.dropout2(x_all, training=training)
        output = self.dense3(x_all)
        return output


# ### 3.2 - Model compiling

# In[ ]:


#Set seed
tf.random.set_seed(71)

#Initialize model
#model = CNN_model(neurons=8, activ='tanh')
model = Hybrid_model(neurons=36, activ='leaky_relu')
#model = Meta_model(neurons=18, activ='tanh')

#Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                          label_smoothing=0.0,
                                          axis=-1,
                                          reduction='sum_over_batch_size',
                                          name='binary_crossentropy')

#Compile the model with loss, optimizer, and metrics
model.compile(loss = loss,
              optimizer = optimizer,
              metrics = [
                  tf.keras.metrics.BinaryAccuracy(),
                  tf.keras.metrics.FalseNegatives(),
                  tf.keras.metrics.FalsePositives(),
                  tf.keras.metrics.TrueNegatives(),
                  tf.keras.metrics.TruePositives()
                  ]
)


# In[ ]:


"""
# Take 1 batch from the dataset and check its content
for batch in train_dataset.take(1):
    (img_batch, meta_batch), target_batch = batch
    
    # Print the shapes of the individual components
    print(f"Image batch shape: {img_batch.shape}")
    print(f"Metadata batch shape: {meta_batch.shape}")
    print(f"Target batch shape: {target_batch.shape}")

# To count the total number of batches
batch_count = 0
for _ in train_dataset:
    batch_count += 1

print(f"Total number of batches in the dataset: {batch_count}")
"""


# ### 3.3 - Model fit

# In[34]:


#Clear the memory leak in Keras
class CustomCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()
    #print(f"Epoch {epoch+1} finished. Validation loss: {logs['val_loss']}")


# In[ ]:


#Set batch sizes
train_batch_size = 32
val_batch_size = 1
test_batch_size = 1

#Determine the number of batches
nb_training_batches = int(len(train_ids_mods)//train_batch_size)
nb_validate_batches = int(len(val_ids_mods)//val_batch_size)
nb_test_batches = int(len(test_ids_mods)//test_batch_size)

#Print results
print("Total training batches in dataset:", nb_training_batches)
print("Total validate batches in dataset:", nb_validate_batches)
print("Total test batches in dataset:", nb_test_batches)


# In[36]:


#Create datasets and get weights
train_dataset, class_weights = make_dataset(hdf5_file, metadata, train_ids_mods, batch_size=train_batch_size, is_training=True, weight_calc=True)
weight_for_0, weight_for_1 = class_weights
validate_dataset, _ = make_dataset(hdf5_file, metadata, val_ids_mods, batch_size = val_batch_size, is_training=False)
test_dataset, _ = make_dataset(hdf5_file, metadata, test_ids_mods, batch_size=test_batch_size)


# In[ ]:


#Run the model through epochs
nb_epochs = 25
early_break = True #End early in case of increasing validation loss

for epoch in range(1, nb_epochs + 1):
    #Make datasets
    np.random.shuffle(train_ids_mods)
    np.random.shuffle(val_ids_mods)
    print("EPOCH", epoch)
    print("First training ID:", train_ids_mods[0][0])
    train_dataset, _ = make_dataset(hdf5_file, metadata, train_ids_mods, batch_size=train_batch_size, is_training=True)
    validate_dataset, _ = make_dataset(hdf5_file, metadata, val_ids_mods, batch_size = val_batch_size, is_training=False)

    mod = model.fit(train_dataset, epochs=1, steps_per_epoch = nb_training_batches, validation_data = validate_dataset, validation_steps = nb_validate_batches, callbacks = [CustomCallback()],
                    class_weight={0: weight_for_0, 1: weight_for_1})
    
    #Save results
    if epoch == 1:
        results = mod.history
    else:
        for key in mod.history:   
            results[key] += mod.history[key]

    #Clean memory after use
    del mod
    del train_dataset
    del validate_dataset
    tf.keras.backend.clear_session()
    gc.collect()

    #Early termination (check after 15 epochs)
    if epoch >= 15 and early_break == True:
        #Calculate previous three changes, if positive, then loss is increasing
        change1 = results["val_loss"][-1] - results["val_loss"][-2]
        change2 = results["val_loss"][-2] - results["val_loss"][-3]
        change3 = results["val_loss"][-3] - results["val_loss"][-4]

        #Three consecutive increases in validation loss will stop the model
        if change1 > 0 and change2 > 0 and change3 > 0:
            break

    #Save occasionally
    #if (epoch % 25 == 0):
    #    model.save(f"XXXX")


# In[ ]:


model.weights


# In[ ]:


#Plot the training and validation losses

#Convert loss results into a datafram
result_preproc = pd.DataFrame({
    'Epoch': [i+1 for i in range(len(results["loss"]))], 
    'Train': results["loss"],
    'Validate': results["val_loss"]
    })

# Convert dataframe from wide to long format
df = pd.melt(result_preproc, ['Epoch'])

#Make plot
g = sns.lineplot(data=df, x='Epoch', y='value', hue='variable')
g.set_title("Loss Curves")
g.legend_.set_title("Loss")
g.set_ylabel('Loss')
g.set_ylim(0, 1)


# **BATCHES**

# In[ ]:


"""
# Iterate through all batches in the dataset and print their shapes
for i, batch in enumerate(train_dataset):
    (img_batch, meta_batch), target_batch = batch
    
    # Print the shapes of the current batch
    print(f"Batch {i+1}:")
    print("  Image Batch Shape:", img_batch.shape)
    print("  Metadata Batch Shape:", meta_batch.shape)
    print("  Target Batch Shape:", target_batch.shape)
"""


# ### 3.4 - Predict Test Data

# In[ ]:


#Retrieve test predictions and real test values
predictions = model.predict(test_dataset, steps = nb_test_batches)
y_pred = np.array([round(i) for i  in predictions.flatten()])
y_test = np.concatenate([y for x, y in test_dataset], axis=0).flatten()
print("Shape of prediction data:", predictions.shape)


# In[42]:


#Calculate the loss
loss = sum(abs(y_test - y_pred))/len(y_pred)


# In[43]:


#Determine true/false positives and negatives
pos_indices = y_test == 1
neg_indices = y_test == 0

#True positives
true_pos = sum(abs(y_test[pos_indices] == y_pred[pos_indices]))

#False negatives
false_neg = sum(abs(y_test[pos_indices] != y_pred[pos_indices]))

#True negatives
true_neg = sum(abs(y_test[neg_indices] == y_pred[neg_indices]))

#False positives
false_pos = sum(abs(y_test[neg_indices] != y_pred[neg_indices]))


# In[ ]:


print("---TEST RESULTS---")
print("Loss on test data:", loss)
print("True positives:", true_pos)
print("False positives:", false_pos)
print("True negatives:", true_neg)
print("False negatives:", false_neg)


#  ## 3.5 Enregistrement du Modele et extinction de l'instance EC2

# In[ ]:


# Function to upload model and results to S3
def upload_to_s3(local_file, bucket_name, s3_file_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket_name, s3_file_key)
        print(f"File {local_file} uploaded to S3 bucket {bucket_name} as {s3_file_key}.")
    except Exception as e:
        print(f"Failed to upload {local_file} to S3: {e}")

# Save the trained model and predictions to S3
def save_model_and_results(model, predictions):
    model_file = "my_model.h5"
    predictions_file = "predictions.npy"
    
    # Save model
    model.save(model_file)
    print(f"Model saved as {model_file}")
    
    # Save predictions (assuming predictions is a numpy array)
    np.save(predictions_file, predictions)
    print(f"Predictions saved as {predictions_file}")
    
    # Upload model and predictions to S3 bucket 'images'
    upload_to_s3(model_file, "images", "models/my_model.h5")
    upload_to_s3(predictions_file, "images", "results/predictions.npy")

# Function to stop EC2 instance after training
def stop_ec2_instance():
    ec2 = boto3.client('ec2', region_name='us-east-1')  # Set the correct region
    instance_id = 'your-instance-id'  # Replace with the actual instance ID
    try:
        ec2.stop_instances(InstanceIds=[instance_id])
        print(f"EC2 instance {instance_id} stopped.")
    except Exception as e:
        print(f"Failed to stop EC2 instance {instance_id}: {e}")

# After model training
# Assuming `model` is your trained model and `predictions` are the results
save_model_and_results(model, predictions)

# Stop EC2 instance after saving results
stop_ec2_instance()

