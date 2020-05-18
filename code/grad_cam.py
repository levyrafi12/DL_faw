import cv2
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
from keras import models
from keras import layers
from keras.models import Model
import tensorflow as tf

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import backend as K
import glob

from data_gen import infest_to_class_ind

def target_category_loss(x, category_ind, n_classes):
    return tf.multiply(x, K.one_hot([category_ind], n_classes))

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def _compute_gradients(tensor, var_list):
	[grads] = tf.gradients(tensor, var_list)
	elems = (var_list, grads)
	grads = tf.map_fn(lambda x: x[1] if x[1] != None else tf.zeros_like(x[0]), \
		elems, dtype=tf.float32)
	return [grads]

def load_and_preprocess_image(img_path, input_dim):
	img = image.load_img(img_path, target_size=input_dim)
	plt.imshow(img)
	plt.show()
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def load_and_transform_image(img_path, input_dim):
	img = cv2.imread(img_path)
	# print(img_fn)
	img = np.array(img) / 255.0
	# plt.imshow(img)
	# plt.show()
	if img.shape[0] < img.shape[1]: # height first
		img = np.transpose(img, (1,0,2))
	# img = np.transpose(img, (1,0,2))
	if img.shape != input_dim:
		img = cv2.resize(img, input_dim[:2])
	# plt.imshow(img)
	# plt.show()
	img = img.reshape((1,) + img.shape)
	return img

def process_images(root_dir, input_dim, image_infest_list, n_classes, n_images=10):
	np.random.seed()
	np.random.shuffle(image_infest_list)

	infests = []
	y = []
	X = []
	n_images = np.minimum(len(image_infest_list), n_images)

	for img_fn, infest in image_infest_list[:n_images]:
		# img = load_and_preprocess_image(img_fn, input_dim)
		img = load_and_transform_image(root_dir + img_fn, input_dim)
		X.append(np.vstack([img]))
		y.append(infest_to_class_ind(n_classes, infest))
		infests.append(infest)
	return X, y

def compute_grad_cam(model, conv_output, loss, image, input_dim):
	grads = normalize(_compute_gradients(loss, conv_output)[0])
	gradient_function = K.function([model.input], [conv_output, grads])
	output, grads_val = gradient_function([image])

	output, grads_val = output[0, :], grads_val[0, :, :, :]
	weights = np.mean(grads_val, axis = (0, 1))
	cam = np.ones(output.shape[0 : 2], dtype = np.float32)

	for i, w in enumerate(weights):
		cam += w * output[:, :, i]

	cam = cv2.resize(cam, input_dim[:2])
	cam = np.maximum(cam, 0)
	heatmap = cam / np.max(cam)

	cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
	cam = np.float32(cam) + np.float32(255 * image[0])
	cam = 255 * cam / np.max(cam)
	cam = np.uint8(cam)
	return cam, heatmap

# gradient class activation maps
def visualize_grad_cam(model, root_dir, n_classes, input_dim, image_infest_list):
	images, labels = process_images(root_dir, input_dim, image_infest_list, n_classes)

	for i, (image, label) in enumerate(zip(images, labels)):
		predictions = model.predict(image)
		[predicted_class] = np.argmax(predictions, axis=1)

		layer_dict = dict([(layer.name, layer) for layer in model.layers])
		final_conv_layer = layer_dict["block5_conv3"]

		conv_output = final_conv_layer.output
		loss = K.sum(target_category_loss(model.layers[-2].output, predicted_class, n_classes))
		cam, _ = compute_grad_cam(model, conv_output, loss, image, input_dim)
		fig=plt.figure(figsize=(224,224))
		fig.suptitle('predicted ' + str(predicted_class) + ' gt ' + str(label) + \
			' infest ' + str(image_infest_list[i][1]))
		orig_img = cv2.imread(root_dir + image_infest_list[i][0])
		fig.add_subplot(1, 3, 1)
		plt.imshow(orig_img)
		plt.title('original')
		fig.add_subplot(1, 3, 2)
		plt.title('resized')
		if orig_img.shape[0] < orig_img.shape[1]:
			image = np.transpose(image, (0,2,1,3))
			cam = np.transpose(cam, (1,0,2))
		plt.imshow(image[0])
		fig.add_subplot(1, 3, 3)
		plt.title('grad-cam')
		plt.imshow(cam)
		plt.show()

	# with tf.Session() as sess:
	# print(sess.run(conv_output))

	# print(model.layers[-1].get_config())
	# layers.get_weights()

	# add(Dense(64, kernel_initializer='random_uniform',bias_initializer='zeros'))