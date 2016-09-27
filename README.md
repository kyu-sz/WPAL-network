# Weakly-supervised Multi-level Attribute Network

By Ken Yu, under guidance of Dr. Zhang Zhang and Prof. Kaiqi Huang.

Weakly-supervised Multi-level Attribute Network (WMA-net) is a Convolutional Neural Network (CNN) structure designed for recognizing attributes from objects. Currently it is developed to recognize attributes from pedestrians only, using the Richly Annotated Pedestrian (RAP) dataset. It is not impossible to adapt this network to recognize attributes of other objects.

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

3. Download the RAP dataset

To get the Richly Annotated Pedestrian (RAP) dataset, please visit rap.idealtest.org to learn about how to download a copy of it.

It should have two zip files.

	```
	$RAP/RAP_annotation.zip
	$RAP/RAP_dataset.zip
	```

4. Unzip them both to the directory.

	```Shell
	cd $RAP
	unzip RAP_annotation.zip
	unzip RAP_dataset.zip
	```

5. Create symlinks for the RAP dataset

	```Shell
	cd $WMA_NET_ROOT/data
	ln -s $RAP RAP
	```

## Usage

To train the model, first fetch a pretrained VGG_CNN_S model by:
	
	```Shell
	./data/scripts/fetch_pretrained_vgg_cnn_s_model.sh
	```

Then run experiment script:

	```Shell
	./experiments/scripts/wma_net.sh 0 VGG_CNN_S \
		--set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"
	```

## Acknowledgements

Some codes are derived from Mr. Ross Girshick's [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).

We use VGG_CNN_S as pretrained model. Information can be found on [Mr. K. Simonyan's Gist](https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9#file-readme-md). It is from the BMVC-2014 paper "Return of the Devil in the Details: Delving Deep into Convolutional Nets":
	```
	Return of the Devil in the Details: Delving Deep into Convolutional Nets
	K. Chatfield, K. Simonyan, A. Vedaldi, A. Zisserman
	British Machine Vision Conference, 2014 (arXiv ref. cs1405.3531)
	```