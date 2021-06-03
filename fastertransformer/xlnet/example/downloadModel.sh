#!/bin/sh
wget https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
mv xlnet_cased_L-12_H-768_A-12 ../data/
mv cased_L-12_H-768_A-12.zip ../data/

wget https://dl.fbaipublicfiles.com/glue/data/STS-B.zip
unzip STS-B.zip
mv STS-B ../data/
mv STS-B.zip ../data/
