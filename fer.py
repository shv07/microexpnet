'''
Title           :fer.py
Description     :functions for testing a sample image or video or live streamed video
Author          :Shivji Bhagat
Date Created    :06-01-2021
Date Modified   :06-01-2021
version         :1.0
python_version  :3.7.8
tensorflow_ver  :2.3.1
'''
#from __future__ import print_function
from time import gmtime, strftime 
from MicroExpNet import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

labels = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# Import the xml files of frontal face and eye

#face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml') 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detectFaces(img):
	# Convertinto grayscale since it works with grayscale images
	gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect the face
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

	if len(faces):
		return faces[0] 
	else:
		return [-13, -13, -13, -13]

# Detects the face and eliminates the rest and resizes the result img
def segmentFace(inputFile, imgXdim, imgYdim):
	# Read the image
	img = cv2.imread(inputFile)  

	# Detect the face
	(p,q,r,s) = detectFaces(img)

	# Return the whole image if it failed to detect the face
	if p != -13: 
		img = img[q:q+s, p:p+r]

	# Crop & resize the image
	img = cv2.resize(img, (256, 256)) 	
	img = img[32:256, 32:256]
	img = cv2.resize(img, (imgXdim, imgYdim)) 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def get_time():
	return strftime("%a, %d %b %Y %X", gmtime())


def predictImage(imagePath:str, modelDir:str):

	#show image
	plt.imshow(cv2.imread(imagePath))
	plt.show()
	
	# Static parameters
	imgXdim = 84
	imgYdim = 84
	nInput = imgXdim*imgYdim # Since RGB is transformed to Grayscale
	
	# Read the image
	image = segmentFace(imagePath, imgXdim, imgYdim)   #return ndarray face from the image - if >1 face only the first one
	testX = np.reshape(image, (1, imgXdim*imgYdim))  #can be modified for more images?
	testX = testX.astype(np.float32)

	# tf Graph input
	x = tf.placeholder(tf.float32, shape=[None, nInput])

	# Construct model
	classifier = MicroExpNet(x)

	# Deploy weights and biases for the model saver
	weights_biases_deployer = tf.train.Saver({"wc1": classifier.w["wc1"], \
										"wc2": classifier.w["wc2"], \
										"wfc": classifier.w["wfc"], \
										"wo": classifier.w["out"],   \
										"bc1": classifier.b["bc1"], \
										"bc2": classifier.b["bc2"], \
										"bfc": classifier.b["bfc"], \
										"bo": classifier.b["out"]})

	with tf.Session() as sess:
		# Initializing the variables
		sess.run(tf.global_variables_initializer())
		#print("[" + get_time() + "] " + "Testing is started...")
		weights_biases_deployer.restore(sess, tf.train.latest_checkpoint(modelDir))
		#print("[" + get_time() + "] Weights & Biases are restored.")				

		predictions = sess.run([classifier.pred], feed_dict={x: testX})
		argmax = np.argmax(predictions)
		print("Name: ", os.path.basename(imagePath))
		print("[" + get_time() + "] Emotion: " + labels[argmax])
		
		sess.close()
	tf.reset_default_graph()

if __name__ == '__main__':
	testDir = "C:\\Users\\shivji\\Desktop\\testImages\\"
	images =[testDir+ i for i in os.listdir(testDir)]

	modelDir = ["E:\\Projects\\FacialExpressionRecognition\\microexpnet\\Models\\CK\\", "E:\\Projects\\FacialExpressionRecognition\\microexpnet\\Models\\OuluCASIA\\"]
	
	#images = ["E:\\Projects\\FacialExpressionRecognition\\CovPoolFER\\conv_feature_pooling\\data\\SFEW_100\\Val\\" + i for i in images]

	for image in images:
		predictImage(image, modelDir[0])
		print("\n\nModel2\n\n")
		#predictImage(image, modelDir[1])
	print("hi")