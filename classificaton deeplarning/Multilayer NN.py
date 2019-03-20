'''******************************************************************************************************************************
Name:- Abhilash G. T														*
Q2:- Multi-layer neural network to identify the type of iris plant from the iris dataset					*
Date:- 17-march-19														*
*********************************************************************************************************************************'''

# Importing Required Libraries
import numpy as np
import pandas as pd
import tensorflow as tf 
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.callbacks import TensorBoard
import time
import matplotlib.pyplot as plt
	


def data_preprocess():
	# Reading the CSV file
	df=pd.read_csv("Your_Path_To_File",names=["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"],)
	global attributes, classes, x_train, x_val, y_train, y_val, x_test, y_test
	# Data preprocessing
	attributes=df.drop('class',axis=1)
	df=pd.get_dummies(df, columns=['class'])
	classes=df.drop(['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm'],axis=1)

	# Normalising the attributes
	norm_attributes = preprocessing.normalize(attributes)

	# Train val split
	x_train, x_val, y_train, y_val = train_test_split(norm_attributes,classes, test_size=0.25, random_state=42)
	# Train test split
	x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size=0.15, random_state=42)

# Neural Network Architecture
def deep_NN():
    global tensorB
    NAME = "tensorboard_Multilayer_NN-{}".format(int(time.time()))
    tensorB = TensorBoard(log_dir='logs/{}'.format(NAME))
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Dense(6,input_dim=attributes.shape[1]))
    model.add(layers.Activation('relu'))
    
    # Hidden Layer 1
    model.add(layers.Dense(12))
    model.add(layers.Activation('relu'))
    
    # Hidden Layer 2
    model.add(layers.Dense(24))
    model.add(layers.Activation('relu'))
    
    # Output Layer
    model.add(layers.Dense(3))
    model.add(layers.Activation('sigmoid'))
    
    return model


def train_model():
	# Compiling the model 
	model.compile(
 	optimizer = "adam",
 	loss = "mean_squared_error",
 	metrics = ["accuracy"],
  	)

	
	# Training the model
	model.fit(x=x_train, y=y_train, batch_size=32, validation_data=(x_val, y_val),epochs=250, callbacks=[tensorB])

	# Finding Test accuracy
	score, acc = model.evaluate(x_test, y_test) 	
	print("Test accuracy =",acc*100)	# Print the test accuracy()

if __name__== "__main__":

	data_preprocess()
	model = deep_NN()  	# Assigning the neural network to model 
	train_model()	  	# Training
	
