# set the matplotlib backend so figures can be saved in the background
#ref https://github.com/Ayushprasad28/Multi-Class-Flower-Classification
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
#from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="training_set" ,
	help="path to input dataset")
ap.add_argument("-m", "--model", type=str, default="trained_model" ,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png" ,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 1
INIT_LR = 1e-5
BS = 1

# initialize the data and labels
data = []
labels = []
dir_labels = ()
num_class = 0

# finding the labels
print("[INFO] Finding Labels...")
for file in os.listdir(args["dataset"]) :
	temp_tuple=(file,'null')
	dir_labels=dir_labels+temp_tuple
	dir_labels=dir_labels[:-1]
	#print(dir_labels)
	num_class=num_class+1
	#print(num_class)

# grab the image paths and randomly shuffle them
print("[INFO] Loading Images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	img = cv2.imread(imagePath)
	height = img.shape[0]
	width = img.shape[1]
	x1 = y1 = 0
	x2 = width
	y2 = height
	diff = abs(height - width)
	if height > width:
		y1 = int(diff/2)
		y2 = height - y1
	elif width>height:
		x1 = int(diff/2)
		x2 = width - x1
	img = img[y1:y2,x1:x2]
	image = cv2.resize(img, (128, 128))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	#print(label,'\n')
	
	for i in range(num_class) :
		if label == dir_labels[i] :
			label = i
	#print(label) 
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.15, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=num_class)
testY = to_categorical(testY, num_classes=num_class)

# construct the image generator for data augmentation
'''
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
'''
#aug = ImageDataGenerator( zoom_range=0.2,
#	horizontal_flip=True, fill_mode="nearest")
# initialize the model
print("[INFO] Compiling Model...")
model = LeNet.build(width=128, height=128, depth=3, classes=num_class)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
model.summary()
# train the network
print("[INFO] Training Network...")
'''
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1 )
'''
# save the model to disk
print("[INFO] Saving Model...")
#model_base=args["model111"]+'.h5'
#model.save(model_base)
#data=pd.DataFrame()
#for i in H.history:
#	np.save(i,np.array(H.history[i]))
#np.save(data,'history')
'''
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
'''
print("[INFO] Completed...")
