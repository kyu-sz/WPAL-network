#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"/pretrained_models
mkdir -p $DIR
cd $DIR

FILE=VGG_CNN_S.caffemodel
URL=http://www.robots.ox.ac.uk/%7Evgg/software/deep_eval/releases/bvlc/$FILE
CHECKSUM=c33d6eb14b3a828224970a894267c516

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading ImageNet-pretrained VGG_CNN_S model..."

wget $URL -O $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
