import cv2
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
from keras import models
from keras import optimizers
from keras import layers
from keras.utils import to_categorical
# from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import argparse

from sklearn.metrics import classification_report

import pandas as pd
from collections import defaultdict, OrderedDict
import os.path
from shutil import copy

from data_gen import DataGenerator, infest_to_class_ind
from grad_cam import visualize_grad_cam

val_ratio = 0.2 # validation data ratio
batch_size = 12
n_epochs = 15
start_epoch = 13
lr = 1e-4
momentum = 0.9
image_dim = (1600, 960, 3) # H * W * channels
input_dim = (512, 512, 3) # after resizing
# input_dim = image_dim
n_classes = 6
min_images_in_set = 5
infest_margin = 0.04
model_name = "vgg_model_512_adam_4_bal_data"
model_suf = ".hdf5"
print_every = 10

root_dir = '../'
img_dir = root_dir + '72159248371744769/2019/0725/'
img_fn = 'uimg_silasche_72159619168141313_2019072509270210.jpg'

reports_file = 'Kenya IPM field reports.xls'
images_file = 'Kenya IPM measurement to photo conversion table.xls'

def parse_cmd():
	parser = argparse.ArgumentParser(description='faw')
	parser.add_argument('--batch-size', type=int, default=12, metavar='N', \
		help='batch size (default: 12)')
	parser.add_argument('--epochs', type=int, default=5, metavar='N', \
		help='number of epochs (default: 5)')
	parser.add_argument('--initial-epoch', type=int, default=0, metavar='N', \
		help='initial epoch (default: 0)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', \
		help='learning rate (default: 0.0001)')
	parser.add_argument('--train', choices=['scratch','freeze', 'unfreeze', 'unfreeze-top'], \
		default='fine-tune', help='set training mechanism: from scratch, freeze cnn layers, \
			unfreeze all cnn layers or unfreeze top cnn layer (default: unfreeze)')
	parser.add_argument('--input-dim', type=int, default=512, metvar=N, \
		help='set image input dimension (default: 512)')
	parser.add_argument('--cnn-base-model', choices=['vgg','resnet'], default='vgg', \
		help='set cnn base model name (default: vgg)')
	parser.add_argument('--classify', choices=['binary','mult'], default='mult', \
		help='set binary or multiple classes (default: mult)')

	return parser.parse_args()

# workaround 
def check_id(i, id):
	# print(i, id)
	id1 = (id // 100) * 100
	# print(id1)
	assert id == id1 or (id - 4) == id1 or (id - 8) == id1 or (id - 96) == id1 or (id - 92) == id1

def fix_id(i, id):
	check_id(i, id)
	return (id + 50) // 100 * 100

def gen_num_image_per_class(image_infest_list):
	class_ind_to_image_num = defaultdict(int)

	for _, infest in image_infest_list:
		class_ind_to_image_num[infest_to_class_ind(n_classes, infest)] += 1

	return sorted(list(class_ind_to_image_num.items()), key=lambda t: t[1])

def balance_data(image_infest_list):
	bal_image_infest_list = []

	for image, infest in image_infest_list:
		class_ind = infest_to_class_ind(n_classes, infest)
		if class_ind in [1,2]:  
			if np.random.uniform() >= 0.25:
				continue
		elif class_ind in [0, 3, 4]:  
			if np.random.uniform() >= 0.75:
				continue
		bal_image_infest_list.append((image, infest))
	return bal_image_infest_list

def split_data(image_infest_list):
	val_size = int(len(image_infest_list) * val_ratio)
	train_data = image_infest_list[:-val_size]
	val_data = image_infest_list[-val_size:]
	return train_data, val_data

def adjust_validation_data(val_data, train_size):
	val_size = int(train_size * val_ratio)
	return val_data[:val_size]

def add_data(data, images, infest):
	for image in images:
		data.append((image, infest))

def prepare_data():
	reports_df = pd.read_excel(root_dir + reports_file, header=None)
	id_infest_list = []
	n_rows = len(reports_df[0][:])

	for i in range(1, n_rows):
		if reports_df[9][i] == 'Seedling':
			continue
		id_infest_list.append((reports_df[0][i], reports_df[7][i]))

	# id_infest_list = list(zip(reports_df[0][1:], reports_df[7][1:]))
	id_infest_list = [(fix_id(i + 2, t[0]), t[1]) for i, t in enumerate(id_infest_list)]
	id_to_infest = dict(id_infest_list)

	images_df = pd.read_excel(root_dir + images_file, header=None)
	id_image_list = list(zip(images_df[0][1:], images_df[1][1:]))
	id_image_list = [(fix_id(i + 2, t[0]), t[1]) for i, t in enumerate(id_image_list)]

	not_found = defaultdict(list) # images not found in reports file

	for i, (key, image) in enumerate(id_image_list):
		# print(i + 2, key, image)
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

	# print(sorted(list(id_to_images_num.items()), key=lambda t: t[0]))

	# id_to_images = sorted(id_to_images.items(), key=lambda t: len(t[1]))

	infest_degree_list = [(0,0)]
	for i in np.arange(1, n_classes):
		low = 1 / (n_classes - 1) * (i - 1) + infest_margin
		high = 1 / (n_classes - 1) * i - infest_margin
		infest_degree_list.append((low, high))

	test_data = []
	last_elem = infest_degree_list[-1] 
	infest_degree_list[-1] = (last_elem[0], 1.01)

	image_infest_list = []
	infest_to_image_num = defaultdict(int)
	np.random.seed(1)

	for i, (id, images) in enumerate(id_to_images.items()):
		if len(images) < min_images_in_set:
			continue
		infest = id_to_infest[id]
		class_ind = infest_to_class_ind(n_classes, infest)
		if n_classes > 2:		
			if infest > 0:
				low, high = infest_degree_list[class_ind]
				if (infest <= low) or (infest >= high):
					add_data(test_data, images, infest)
					continue
			if infest < 0.1:
				if np.random.uniform() >= 0.1:
					for image in images:
						add_data(test_data, images, infest)
					continue
		for image in images:
			image_infest_list.append((image, infest))
			infest_to_image_num[infest] += 1

	# print(sorted(list(infest_to_image_num.items()), key=lambda t: t[1]))
	print("before balancing #{} {}".format(len(image_infest_list), \
		gen_num_image_per_class(image_infest_list)))
	# copy(root_dir + images[0], "images_dir/")
	# print(key, images[0], id_to_infest[key])

	np.random.shuffle(image_infest_list)
	train_data, val_data = split_data(image_infest_list)
	train_data = balance_data(train_data)
	# image_infest_list = image_infest_list[:len(train_data)] # // 6
	print("after balancing #{} {}".format(len(train_data), \
		gen_num_image_per_class(train_data)))

	for i, (image_fn, label) in enumerate(image_infest_list):
		dest_image = "images_dir/pict_" + str(i) + "_" + \
		str(infest_to_class_ind(n_classes, infest)) + ".jpg"
		# copy(root_dir + image_fn,  dest_image)

	val_data = adjust_validation_data(val_data, len(train_data))

	return train_data, val_data, val_data

def set_optimize(model):
	model.compile(optimizer=optimizers.Adam(lr=lr), \
		loss='categorical_crossentropy', metrics=['acc'])

def set_trainable(model, trainable):
	for layer in model.layers:
		layer.trainable = trainable

def build_vgg_model(freeze=True):
	conv_base = VGG16(include_top=False, weights='imagenet', input_shape=input_dim)
	# model.add(layers.MaxPooling2D(pool_size=(2,2)))
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_base.output)
	x = layers.Flatten()(x)
	x = layers.Dense(2048, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1024, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	# model.add(layers.Dense(n_classes, activation='softmax'))
	x = layers.Dense(n_classes)(x) 
	last = layers.Activation('softmax')(x)
	model = Model(inputs=conv_base.input, outputs=last)

	if freeze:
		set_trainable(conv_base, False)
	set_optimize(model)

	return model, conv_base

def build_resnet_model(freeze=False):
	conv_base = ResNet50(include_top=False, weights=None, input_shape=input_dim)	
	x = layers.MaxPooling2D(pool_size=(4,4))(conv_base.output)
	x = layers.Flatten()(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(2048, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(1024, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(n_classes)(x) 
	last = layers.Activation('softmax')(x)
	model = Model(inputs=conv_base.input, outputs=last)

	if freeze:
		set_trainable(conv_base, False)
	set_optimize(model)

	return model, conv_base

def get_next_batch(image_infest_list, batch_size=1):
	np.random.seed(1)
	np.random.shuffle(image_infest_list)
	n_samples = len(image_infest_list)

	x_vecs = []
	y_labels = []
	orig_data = []
	for i, (img_fn, infest) in enumerate(image_infest_list):
		orig_img = cv2.imread(root_dir + img_fn)
		# print(root_dir + img_fn)
		img = np.array(orig_img) / 255.0
		if img.shape[0] < img.shape[1]: # height first
			img = np.transpose(img, (1,0,2))
		if img.shape != input_dim:
			img = cv2.resize(img, (input_dim[0], input_dim[1]))
		# plt.imshow(img)
		# plt.show()
		y_labels.append(infest_to_class_ind(n_classes, infest))
		x_vecs.append(img.reshape((1,) + img.shape))
		orig_data.append((orig_img, infest))
		if len(x_vecs) % batch_size == 0 or i == n_samples - 1:
			y_labels = to_categorical(y_labels, n_classes)
			x_vecs = np.vstack(x_vecs)
			yield x_vecs, y_labels, orig_data
			x_vecs = []
			y_labels = []
			orig_data = []

def predict(model, data):
	print("Pediction: #samples {}".format(len(data)))

	n_match = 0
	n_batch = len(data) // batch_size
	n_batch = n_batch if (len(data) % batch_size) == 0 else n_batch + 1
	count_miss_pred = [0] * n_classes
	count_pred = [0] * n_classes
	count_gt = [0] * n_classes
	count_hit_pred = [0] * n_classes

	for i, (images, one_hot_labels, orig_data) in enumerate(get_next_batch(data, batch_size)):
		if (i + 1) % print_every == 0:
			print("Iteration {}/{}".format(i + 1, n_batch))
		predictions = model.predict(images)
		predictions = predictions.argmax(axis=1)
		labels = np.argmax(one_hot_labels, axis=1)
		n_match += np.sum(y_pred == y_true for y_pred, y_true in zip(predictions, labels))
		for y_pred, y_true, (orig_img, infest) in zip(predictions, labels, orig_data):
			if y_pred != y_true:
				print("pred {} gt {} infest {}".format(y_pred, y_true, infest))
				# plt.imshow(orig_img)
				# plt.show()
				count_miss_pred[y_pred] += 1
			else:
				count_hit_pred[y_pred] += 1
			count_pred[y_pred] += 1
			count_gt[y_true] += 1

	print("Test acc {}".format(n_match / len(data)))
	print("count miss preds {}".format(count_miss_pred))
	print("count hit preds {}".format(count_hit_pred))
	print("count preds {}".format(count_pred))
	print("count gt {}".format(count_gt))

def evaluate(model, data):
	# evaluate the network
	print("Evaluating: #samples {}".format(len(data)))
	generator = DataGenerator(root_dir, data, batch_size, input_dim, n_classes)
	_, infests = zip(*generator.image_infest_list)
	labels = [infest_to_class_ind(n_classes, infest) for infest in infests]

	predictions = model.predict_generator(generator, verbose=True)
	predictions = predictions.argmax(axis=1)
	n_match = np.sum(y_pred == y_true for y_pred, y_true in zip(predictions, labels))

	print("Test acc {}".format(n_match / len(labels)))
	print(classification_report(labels, predictions))

def train_and_eval(model, train_gen, val_gen):
	print("fit_generator: #training {}, #validation {}"\
		.format(len(train_gen.image_infest_list), len(val_gen.image_infest_list)))

	checkpoint = ModelCheckpoint(model_name + model_suf)

	H = model.fit_generator(generator=train_gen, validation_data=val_gen, \
		epochs=n_epochs, initial_epoch=start_epoch, callbacks=[checkpoint])

	return H

def train_or_eval(model, image_infest_list, train=True):
	epochs = n_epochs if train else 1
	str = "training" if train else "validation"
	print("train_on_batch: num {} keys {}".format(str, len(image_infest_list)))

	for epoch in range(1, epochs + 1):
		print("epoch {}".format(epoch))
		for i, (x, infest, _) in enumerate(get_next_batch(image_infest_list, batch_size)):
			y = infest_to_class_ind(n_classes, infest)
			print("iter {}".format(i + 1))
			if train:
				ret = model.train_on_batch(x, y)
			else:
				ret = model.evaluate(x, y, batch_size)

			print("loss {} acc {}".format(ret[0], ret[1]))

	# plt.imshow(img[0])
	# plt.show()id_to_images, id_to_infest, train_keys, val_keys

def data_generator(train_data, val_data):
	train_gen = DataGenerator(root_dir, train_data, batch_size, input_dim, n_classes)
	val_gen = DataGenerator(root_dir, val_data, batch_size, input_dim, n_classes)

	return train_gen, val_gen

def unfreeze_vgg_final_block(model, layer_dict):
	for i in range(1,4):
		name = "block5_conv" + str(i)
		layer = layer_dict[name]
		layer.trainable = True

	set_optimize(model)

def create_or_load_model(build_func, unfreeze_final_block=None, unfreeze=False, update_lr=False):
	if os.path.isfile(model_name + model_suf):
		model = models.load_model(model_name + model_suf)
		if unfreeze_final_block != None:
			layer_dict = dict([(layer.name, layer) for layer in model.layers])
			unfreeze_final_block(model, layer_dict)
		elif unfreeze:
			set_trainable(model, True)
			set_optimize(model)
		elif update_lr:
			set_optimize(model)
		# print(model.get_config())

	else:
		model, _ = build_func()

	model.summary()
	print(model.optimizer.get_config())
	print('iterations is ', K.get_session().run(model.optimizer.iterations))
	return model

def plot_loss_or_acc(str, H, graph_name=model_name):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(start_epoch, n_epochs), H.history[str], label="train_" + str)
	plt.plot(np.arange(start_epoch, n_epochs), H.history["val_" + str], label="val_" + str)
	plt.title("Training " + str +" on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel(str)
	plt.legend(loc="lower left")
	plt.savefig("graph_" + graph_name + "_" + str)
	plt.clf()

def plot_graph(H):
	plot_loss_or_acc("loss", H)
	plot_loss_or_acc("acc", H)

def faw():
	train_data, val_data, test_data = prepare_data()
	train_gen, val_gen = data_generator(train_data, val_data)
	model = create_or_load_model(build_vgg_model)
	# visualize_grad_cam(model, root_dir, n_classes, input_dim, train_data + val_data)
	H = train_and_eval(model, train_gen, val_gen)
	plot_graph(H)
	# evaluate(model, test_data)
	# predict(model, test_data)
	# train_or_eval(model, id_t# images, id_to_infest, train_keys)
	# train_or_eval(model, id_to_images, id_to_infest, val_keys, False)

if __name__ == '__main__':
	faw()