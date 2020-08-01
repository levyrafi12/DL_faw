import cv2
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import argparse

from sklearn.metrics import classification_report, auc, roc_curve
from sklearn.utils import class_weight

import pandas as pd
from collections import defaultdict, OrderedDict
import os.path
from shutil import copy

from data_gen import DataGenerator, infest_to_class_ind
from grad_cam import visualize_grad_cam
from faw_utils import plot_ROC, plot_histogram, \
	plot_histogram_multiclasses, gen_num_image_per_class

## user defined parameters
# 'unfreeze' - if False freeze the weights of VGG16 model, otherwise unfreeze
# the weights of the final CONV block If the model does not exist, 
# weights will always be freezed
unfreeze = True 
training = True # if False, do prediction only
gradCAM = False # if True, visualize Grad-CAM heatmaps (model must be trained)
n_classes = 6 # Ran on 2 or 6 classes 
batch_size = 32 
start_epoch = 0 # starting epoch 
n_epochs = 30 # num epochs 
val_ratio = 0.2 # validation data ratio
# model_name = "vgg_model_bs_32_bin" # healthy vs infested model
model_name = "vgg_model_bs_32_nc6" # infested severity classification model (6 classes)
#### end user defined parameters

lr = 1e-5 # initial learning rate 
momentum = 0.9
fc_layer_size = 2048
image_dim = (1600, 960, 3) # H * W * channels
target_dim = (512, 512, 3) # after resizing
min_images_in_set = 5 # Only fields having 5 or more images will be considered
infest_margin = 0.04

model_suf = ".hdf5"

root_dir = '../'
img_dir = root_dir + '72159248371744769/2019/0725/'
# img_fn = 'uimg_silasche_72159619168141313_2019072509270210.jpg'

reports_file = 'Kenya IPM field reports.xls'
images_file = 'Kenya IPM measurement to photo conversion table.xls'

# workaround 
def check_id(i, id):
	# print(i, id)
	id1 = (id // 100) * 100
	# print(id1)
	assert id == id1 or (id - 4) == id1 or (id - 8) == id1 or (id - 96) == id1 or (id - 92) == id1

def fix_id(i, id):
	check_id(i, id)
	return (id + 50) // 100 * 100

def split_data(image_infest_list):
	val_size = int(len(image_infest_list) * val_ratio)
	train_data = image_infest_list[:-val_size]
	val_data = image_infest_list[-val_size:]
	return train_data, val_data

def prepare_data():
	reports_df = pd.read_excel(root_dir + reports_file, header=None)
	id_infest_list = []
	n_rows = len(reports_df[0][:])

	for i in range(1, n_rows):
		if reports_df[9][i] == 'Seedling':
			continue
		id_infest_list.append((reports_df[0][i], reports_df[7][i]))

	id_infest_list = [(fix_id(i + 2, t[0]), t[1]) for i, t in enumerate(id_infest_list)]
	id_to_infest = dict(id_infest_list)

	images_df = pd.read_excel(root_dir + images_file, header=None)
	id_image_list = list(zip(images_df[0][1:], images_df[1][1:]))
	id_image_list = [(fix_id(i + 2, t[0]), t[1]) for i, t in enumerate(id_image_list)]

	not_found = defaultdict(list) # images not found in reports file

	for i, (key, image) in enumerate(id_image_list):
		if id_to_infest.get(key) == None:
			# print("File not reported {} {} {}".format(i + 2, key, image))
			not_found[key].append(i + 2)

	id_to_images = defaultdict(list)
	for id, img_fn in id_image_list:
		if not_found.get(id) == None:
			if os.path.isfile(root_dir + img_fn):
				id_to_images[id].append(img_fn)
			# else:
				# print("File not found {}".format(root_dir + img))

	# print(not_found)
	# print("num missing images {}".format(sum([len(t[1]) for t in not_found.items()])))

	id_to_images_num = defaultdict(int)

	for _, val in id_to_images.items():
		id_to_images_num[len(val)] += 1

	infest_degree_list = [(0,0)]
	for i in np.arange(1, n_classes - 1):
		low = 1 / (n_classes - 1) * (i - 1) + infest_margin
		high = 1 / (n_classes - 1) * i - infest_margin
		infest_degree_list.append((low, high))

	last_elem = infest_degree_list[-1]
	delta = 2 * infest_margin if n_classes > 2 else 0
	infest_degree_list.append((last_elem[1] + delta, 1.01))
	test_data = []

	image_infest_list = []
	np.random.seed(1)

	for i, (id, images) in enumerate(id_to_images.items()):
		# requiring a minimum number samples per field (set to 5)
		if len(images) < min_images_in_set:
			continue
		infest = id_to_infest[id]
		class_ind = infest_to_class_ind(n_classes, infest)
		if n_classes > 2:		
			if infest > 0:
				low, high = infest_degree_list[class_ind]
				if (infest < low) or (infest > high):
					for image in images:
						# leave aside 10% of the data for testing
						if np.random.uniform() <= 0.1:	
							test_data.append((image, infest))
					continue
		for image in images:
			# leave aside 10% of the data for testing
			if np.random.uniform() <= 0.1:
				test_data.append((image, infest))
			else:
				image_infest_list.append((image, infest))

	np.random.shuffle(image_infest_list)
	print("data before splitting #{} {}".format(len(image_infest_list), \
		gen_num_image_per_class(image_infest_list, n_classes)))
	train_data, val_data = split_data(image_infest_list)
	print("trained data after splitting #{} {}".format(len(train_data),
		gen_num_image_per_class(train_data, n_classes)))

	# plot_histogram(image_infest_list)
	# lot_histogram_multiclasses(image_infest_list)
	
	return train_data, val_data, test_data

def set_optimize(model):
	# model.compile(optimizer=optimizers.Adam(lr=lr), 
	model.compile(optimizer=optimizers.Adam(lr=lr), \
		loss='categorical_crossentropy', metrics=['acc'])

def set_trainable(model, trainable):
	for layer in model.layers:
		layer.trainable = trainable

def scheduler(epoch, lr):
	if epoch < 10:
		return 1e-3
	elif epoch < 20: 
 		return 1e-4
	return 1e-5 

def build_vgg_model():
	conv_base = VGG16(include_top=False, weights='imagenet', input_shape=target_dim)
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_base.output)
	x = layers.Flatten()(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(fc_layer_size, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(fc_layer_size // 2, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(n_classes)(x) 
	last = layers.Activation('softmax')(x)
	model = Model(inputs=conv_base.input, outputs=last)

	set_trainable(conv_base, False)
	set_optimize(model)

	return model, conv_base

def predict(model, data):
	print("Prediction: #samples {}".format(len(data)))
	generator = DataGenerator(root_dir, model, data, batch_size, target_dim, n_classes)
	_, infests = zip(*generator.image_infest_list)
	labels = [infest_to_class_ind(n_classes, infest) for infest in infests]

	predictions = model.predict_generator(generator, verbose=True)
	label_predictions = predictions.argmax(axis=1)
	n_match = sum(y_pred == y_true for y_pred, y_true in zip(label_predictions, labels))

	if n_classes > 2:
		fpr, tpr, _ = roc_curve(labels, predictions[:,1])
		auc_keras = auc(fpr, tpr)
		plot_ROC(fpr, tpr, auc_keras)

	print("Test acc {}".format(n_match / len(labels)))
	print(classification_report(labels, label_predictions))

def train_and_eval(model, train_gen, val_gen):
	_, y_train = zip(*train_gen.image_infest_list)
	y_train = [infest_to_class_ind(n_classes, y) for y in y_train]
	class_weights = class_weight.compute_class_weight('balanced',\
		np.unique(y_train), y_train)

	print("fit_generator: #training {}, #validation {}"\
		.format(len(train_gen.image_infest_list), len(val_gen.image_infest_list)))

	checkpoint = ModelCheckpoint(model_name + model_suf, \
		save_best_only=True, monitor='val_acc')

	lr_schedular = LearningRateScheduler(scheduler)
	early_stopping = EarlyStopping(monitor='loss', patience=3)

	H = model.fit_generator(generator=train_gen, validation_data=val_gen, \
		epochs=n_epochs, initial_epoch=start_epoch, \
		class_weight=class_weights,
		callbacks=[checkpoint, lr_schedular, early_stopping])

	return H

def data_generator(model, train_data, val_data):
	train_gen = DataGenerator(root_dir, model, train_data, batch_size, target_dim, n_classes)
	val_gen = DataGenerator(root_dir, model, val_data, batch_size, target_dim, n_classes)

	return train_gen, val_gen

def unfreeze_final_block(model, layer_dict):
	for i in range(1,4): # layer within a block
		for j in range(5,6): # block ind
			name = "block" + str(j) + "_conv" + str(i)
			layer = layer_dict[name]
			layer.trainable = True

	set_optimize(model)

def create_or_load_model(build_func):
	if os.path.isfile(model_name + model_suf):
		model = models.load_model(model_name + model_suf)
		if unfreeze == True:
			layer_dict = dict([(layer.name, layer) for layer in model.layers])
			unfreeze_final_block(model, layer_dict)
	else:
		model, _ = build_func()

	model.summary()
	print(model.optimizer.get_config())
	print(K.eval(model.optimizer.lr))
	t = K.cast(model.optimizer.iterations, K.floatx())
	print("iterations {}".format(t))
	return model

def plot_loss_or_acc(metric, H, graph_name=model_name):
	plt.style.use("ggplot")
	plt.figure()
	end_epoch = start_epoch + len(H.history[metric])
	plt.plot(np.arange(start_epoch, end_epoch), H.history[metric], label="train_" + metric)
	plt.plot(np.arange(start_epoch, end_epoch), H.history["val_" + metric], label="val_" + metric)
	plt.title("Training " + metric + " on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel(metric)
	plt.legend(loc="lower left")
	plt.savefig("graph_" + graph_name + "_" + \
		str(start_epoch + 1) + "_" +  str(end_epoch) + "_" + metric)
	plt.clf()

def plot_graph(H):
	plot_loss_or_acc("loss", H)
	plot_loss_or_acc("acc", H)

def faw():
	init_gpus()
	model = create_or_load_model(build_vgg_model)
	train_data, val_data, test_data = prepare_data()
	if gradCAM == True:
		visualize_grad_cam(model, root_dir, n_classes, target_dim, train_data + val_data)
	if training == True:
		train_gen, val_gen = data_generator(model, train_data, val_data)
		H = train_and_eval(model, train_gen, val_gen)
		plot_graph(H)
	predict(model, test_data)

def init_gpus():
	# tf.debugging.set_log_device_placement(True)
	gpus = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_visible_devices(gpus[5:6], 'GPU')

if __name__ == '__main__':
	faw()
	
