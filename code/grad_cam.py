import cv2
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16
import glob
import imutils

from data_gen import infest_to_class_ind

dest_dir = 'grad_cam_dir'

def load_and_transform_image(img_path, target_dim):
	img = cv2.imread(img_path)
	img = np.array(img) / 255.0
	if img.shape[0] < img.shape[1]: # height first
		img = np.transpose(img, (1,0,2))
	img = cv2.resize(img, target_dim[:2])
	img = img.reshape((1,) + img.shape)
	return img

def process_images(root_dir, target_dim, image_infest_list, n_classes, n_images=100):
	np.random.seed()
	np.random.shuffle(image_infest_list)

	infests = []
	y = []
	X = []
	n_images = np.minimum(len(image_infest_list), n_images)

	for img_fn, infest in image_infest_list[:n_images]:
		img = load_and_transform_image(root_dir + img_fn, target_dim)
		X.append(np.vstack([img]))
		y.append(infest_to_class_ind(n_classes, infest))
		infests.append(infest)
	return X, y

def compute_grad_cam(tape, model, conv_output, loss, image, target_dim):
	grads = tape.gradient(loss, conv_output)
	# compute the guided gradients
	cast_conv_output = tf.cast(conv_output > 0, "float32")
	cast_grads = tf.cast(grads > 0, "float32")
	guided_grads = cast_conv_output * cast_grads * grads

	# the convolution and guided gradients have a batch dimension
	# (which we don't need) so let's grab the volume itself and
	# discard the batch
	conv_output = conv_output[0]
	guided_grads = guided_grads[0]

	# compute the average of the gradient values, and using them
	# as weights, compute the ponderation (weight) of the filters with
	# respect to the weightsa
	weights = tf.reduce_mean(guided_grads, axis=(0, 1))
	cam = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)
	cam = cv2.resize(cam.numpy(), target_dim[:2])
	cam = np.maximum(cam, 0)
	heatmap = cam / np.max(cam)
	heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
	cam = np.float32(heatmap) + np.float32(255 * image[0])
	cam = 255 * cam / np.max(cam)
	heatmap = (heatmap * 255).astype("uint8")
	cam = np.uint8(cam)
	return heatmap, cam

# gradient class activation maps
def visualize_grad_cam(model, root_dir, n_classes, target_dim, image_infest_list):
	images, labels = process_images(root_dir, target_dim, image_infest_list, n_classes)

	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	final_conv_layer = layer_dict["block5_conv3"]

	gradModel = Model(
		inputs=[model.inputs],
		outputs=[final_conv_layer.output, model.output])

	for i, (image, label) in enumerate(zip(images, labels)):
		with tf.GradientTape() as tape:
			inputs = tf.cast(image, tf.float32)
			(conv_output, predictions) = gradModel(inputs)
			loss = predictions[:, label]
		[predicted_class] = np.argmax(predictions, axis=1)
		_, cam = compute_grad_cam(tape, model, conv_output, loss, image, target_dim)
		text = 'predicted ' + str(predicted_class) + ' gt ' + str(label) + \
			' infest ' + str(image_infest_list[i][1])
		orig_img = cv2.imread(root_dir + image_infest_list[i][0])
		if orig_img.shape[0] < orig_img.shape[1]:
			image = np.transpose(image, (0,2,1,3))
			cam = np.transpose(cam, (1,0,2))
		image = (image[0] * 255).astype("uint8")
		h_img = cv2.hconcat([image, cam])
		h_img = imutils.resize(h_img, width=600)
		cv2.imshow(text, h_img)
		# cv2.imwrite(dest_dir + 'cam_' + str(i) + '_' + text + '.jpg', h_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
