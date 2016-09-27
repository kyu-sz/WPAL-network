# Weakly-supervised Multi-level Attribute Network

By Ken Yu, under guidance of Dr. Zhang Zhang and Prof. Kaiqi Huang.

Weakly-supervised Multi-level Attribute Network is a Convolutional Neural Network (CNN) structure designed for recognizing attributes from objects. Currently it is developed to recognize attributes from pedestrians only, using the Richly Annotated Pedestrian (RAP) dataset. It is not impossible to adapt this network to recognize attributes of other objects.

## Installation

1. Clone this repository

	```Shell
	# Make sure to clone with --recursive
	git clone --recursive https://github.com/kyu-sz/Weakly-supervised-Multi-level-Attribute-Network.git
	```

2. Build Caffe and pycaffe

	This project use python layers for input, etc. When building Caffe, set the WITH_PYTHON_LAYER option to true.

	```Shell
	WITH_PYTHON_LAYER=1 make all pycaffe -j 8
	```

## Acknowledgements

Some codes are derived from Ross Girshick's [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).