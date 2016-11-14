import math

import cv2
import numpy as np
from utils.blob import img_list_to_blob

from config import cfg


def _get_image_blob(img):
	"""Converts an image into a network input.
	Arguments:
		img (ndarray): a color image in BGR order
	Returns:
		blob (ndarray): a data blob holding the image
		img_scale (double): image scale (relative to img) used
	"""
	img_orig = img.astype(np.float32, copy=True)
	img_orig -= cfg.PIXEL_MEANS

	img_shape = img_orig.shape
	img_size_min = np.min(img_shape[0:2])
	img_size_max = np.max(img_shape[0:2])

	processed_images = []

	target_size = cfg.TEST.SCALE
	img_scale = float(target_size) / float(img_size_max)
	# Prevent the biggest axis from being more than MAX_SIZE
	if np.round(img_scale * img_size_min * img_scale * img_size_max) > cfg.TEST.MAX_AREA:
		img_scale = math.sqrt(float(cfg.TEST.MAX_AREA) / float(img_size_min * img_size_max))
	# Prevent the shorter sides from being less than MIN_SIZE
	if np.round(img_scale * img_size_min < cfg.MIN_SIZE):
		img_scale = np.round(cfg.MIN_SIZE / img_size_min) + 1

	img = cv2.resize(img_orig, None, None, fx=img_scale, fy=img_scale,
	                 interpolation=cv2.INTER_LINEAR)
	processed_images.append(img)

	# Create a blob to hold the input images
	blob = img_list_to_blob(processed_images)

	return blob, img_scale


def _get_blobs(im):
	"""Convert an image into network inputs."""
	blobs = {'data': None}
	blobs['data'], img_scale_factor = _get_image_blob(im)
	return blobs, img_scale_factor


def _attr_group_norm(pred, group):
	for i in group:
		pred[i] = 1 if pred[i] == max(pred[group]) else 0
	return pred


def discretize(attr, threshold):
	for i in xrange(attr.shape[0]):
		attr[i] = 0 if attr[i] < threshold[i] else 1


def recognize_attr(net, img, attr_group, threshold=None):
	"""Recognize attributes in a pedestrian image.

	Arguments:
		net (caffe.Net): WMA network to use.
		img (ndarray): color image to test (in BGR order)
		attr_group(list of ranges): a list of ranges, each contains indexes
									of attributes that mutually exclude each other.

	Returns:
		attributes (ndarray): K x 1 array of predicted attributes. (K is
			specified by database or the net)
	"""
	blobs, img_scale = _get_blobs(img)

	# reshape network inputs
	net.blobs['data'].reshape(*(blobs['data'].shape))

	forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
	blobs_out = net.forward(**forward_kwargs)

	pred = np.average(blobs_out['pred'], axis=0)
	heat3 = np.average(blobs_out['heat3'], axis=0)
	heat4 = np.average(blobs_out['heat4'], axis=0)
	heat5 = np.average(blobs_out['heat5'], axis=0)
	score = np.average(blobs_out['score'], axis=0)

	for group in attr_group:
		pred = _attr_group_norm(pred, group)

	if threshold is not None:
		for i in xrange(pred.shape[0]):
			pred[i] = 0 if pred[i] < threshold[i] else 1

	return pred, heat3, heat4, heat5, score, img_scale
