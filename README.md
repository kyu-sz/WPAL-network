# Weakly-supervised Multi-level Attribute Network

By Ken Yu, under guidance of Dr. Zhang Zhang and Prof. Kaiqi Huang.

Weakly-supervised Multi-level Attribute is a Convolutional Neural Network (CNN) structure designed for recognizing attributes from objects. Currently it is developed to recognize attributes from pedestrians only, using the Richly Annotated Pedestrian (RAP) dataset. It is not impossible to adapt this network to recognize attributes of other objects.

This project use python layers for input, etc. When building Caffe, set WITH_PYTHON_LAYER option to true.

```Shell
WITH_PYTHON_LAYER=1 make all pycaffe
```

Some codes are written imitating Ross Girshick's [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).
