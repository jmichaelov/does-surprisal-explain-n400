#!/bin/bash

mkdir data

wget -O data/graph-2016-09-10.pbtxt http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt

wget -O data/ckpt-base http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base
wget -O data/ckpt-char-embedding http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding
wget -O data/ckpt-lstm http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm
wget -O data/ckpt-softmax0 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0
wget -O data/ckpt-softmax1 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1
wget -O data/ckpt-softmax2 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2
wget -O data/ckpt-softmax3 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3
wget -O data/ckpt-softmax4 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4
wget -O data/ckpt-softmax5 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5
wget -O data/ckpt-softmax6 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6
wget -O data/ckpt-softmax7 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7
wget -O data/ckpt-softmax8 http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8

wget -O data/vocab-2016-09-10.txt http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt
wget -O data/news.en.heldout-00000-of-00050 http://download.tensorflow.org/models/LM_LSTM_CNN/test/news.en.heldout-00000-of-00050
