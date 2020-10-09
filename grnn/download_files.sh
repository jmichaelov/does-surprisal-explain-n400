#!/bin/bash

mkdir -p data/lm/English

wget -O data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt
wget -O data/lm/English/vocab.txt https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt
