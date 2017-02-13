import os
import json
import urllib

import h5py
import numpy as np
import pickle as pk

from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file

# Load models and support
first_gate = VGG16(weights='imagenet')
print "First gate loaded"
second_gate = load_model('static/models/d1_ft_model.h5')
print "Second gate loaded"
location_model = load_model('static/models/d2_ft_model.h5')
print "Location model loaded"
severity_model = load_model('static/models/d3_ft_model.h5')
print "Severity model loaded"
with open('static/models/vgg16_cat_list.pk', 'rb') as f:
	cat_list = pk.load(f)
print "Cat list loaded"

# from Keras GitHub  
CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def get_predictions(preds, top=5):
	global CLASS_INDEX
	if len(preds.shape) != 2 or preds.shape[1] != 1000:
		raise ValueError('`decode_predictions` expects '
						 'a batch of predictions '
						 '(i.e. a 2D array of shape (samples, 1000)). '
						 'Found array with shape: ' + str(preds.shape))
	if CLASS_INDEX is None:
		fpath = get_file('imagenet_class_index.json',
						 CLASS_INDEX_PATH,
						 cache_subdir='models')
		CLASS_INDEX = json.load(open(fpath))
	l = []
	for pred in preds:
		top_indices = pred.argsort()[-top:][::-1]
		indexes = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
		indexes.sort(key=lambda x: x[2], reverse=True)
		l.append(indexes)
	return l

def prepare_img_224(img_path):
	img = load_img(img_path, target_size=(224, 224))
	x = img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def car_categories_gate(img_224, model):
	print "Validating that this is a picture of your car..."
	out = model.predict(img_224)
	top = get_predictions(out, top=5)
	for j in top[0]:
		if j[0:2] in cat_list:
			# print j[0:2]
			return True 
	return False

def prepare_img_256(img_path):
	img = load_img(img_path, target_size=(256, 256)) # this is a PIL image 
	x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
	x = x.reshape((1,) + x.shape)/255
	return x

def car_damage_gate(img_256, model):
	print "Validating that damage exists..."
	pred = model.predict(img_256)
	if pred[0][0] <=.5:
		return True # print "Validation complete - proceed to location and severity determination"
	else:
		return False
		# print "Are you sure that your car is damaged? Please submit another picture of the damage."
		# print "Hint: Try zooming in/out, using a different angle or different lighting"

def location_assessment(img_256, model):
	print "Determining location of damage..."
	pred = model.predict(img_256)
	pred_label = np.argmax(pred, axis=1)
	d = {0: 'Front', 1: 'Rear', 2: 'Side'}
	for key in d.iterkeys():
		if pred_label[0] == key:
			return d[key]
	# 		print "Assessment: {} damage to vehicle".format(d[key])
	# print "Location assessment complete."

def severity_assessment(img_256, model):
	print "Determining severity of damage..."
	pred = model.predict(img_256)
	pred_label = np.argmax(pred, axis=1)
	d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
	for key in d.iterkeys():
		if pred_label[0] == key:
			return d[key]
	# 		print "Assessment: {} damage to vehicle".format(d[key])
	# print "Severity assessment complete."

# load models
def engine(img_path):
	img_224 = prepare_img_224(img_path)
	g1 = car_categories_gate(img_224, first_gate)

	if g1 is False:
		result = {'gate1': 'Car validation check: ', 
		'gate1_result': 0, 
		'gate1_message': {0: 'Are you sure this is a picture of your car? Please retry your submission.', 
		1: 'Hint: Try zooming in/out, using a different angle or different lighting'},
		'gate2': None,
		'gate2_result': None,
		'gate2_message': {0: None, 1: None},
		'location': None,
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
		return result
		
	img_256 = prepare_img_256(img_path)
	g2 = car_damage_gate(img_256, second_gate)

	if g2 is False:
		result = {'gate1': 'Car validation check: ', 
		'gate1_result': 1, 
		'gate1_message': {0: None, 1: None},
		'gate2': 'Damage presence check: ',
		'gate2_result': 0,
		'gate2_message': {0: 'Are you sure that your car is damaged? Please retry your submission.',
		1: 'Hint: Try zooming in/out, using a different angle or different lighting.'},
		'location': None,
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
		return result
	
	x = location_assessment(img_256, location_model)
	y = severity_assessment(img_256, severity_model)
	
	result = {'gate1': 'Car validation check: ', 
	'gate1_result': 1, 
	'gate1_message': {0: None, 1: None},
	'gate2': 'Damage presence check: ',
	'gate2_result': 1,
	'gate2_message': {0: None, 1: None},
	'location': x,
	'severity': y,
	'final': 'Damage assessment complete!'}
	return result