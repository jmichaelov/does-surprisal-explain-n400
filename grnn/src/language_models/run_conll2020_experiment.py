# This version Copyright (c) 2020 James A. Michaelov. 
# All Rights Reserved
# All changes from original file ('evaluate_target_word.py') licensed under the same license as the original file (CC BY-NC 4.0)

# Original license note below: 

    
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

parser = argparse.ArgumentParser(description='Mask-based evaluation: extracts softmax vectors for specified words')

parser.add_argument('--data', type=str,
                    help='location of the data corpus for LM training')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--in_file','-i', type=str, help='input file')
parser.add_argument('--out_dir','-o', type=str, help='output directory')
args = parser.parse_args()


def get_directory(stims_file):
    directory_divided = stims_file.split('/')
    directory = '/'.join(directory_divided[:-1])
    filename = directory_divided[-1]
    filename_split = filename.split('.')
    filename_name = filename_split[0]
    filename_ext = filename_split[1]
    return([directory,filename_name,filename_ext])
        
def evaluate(data_source, mask):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0

    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            # keep continuous hidden state across all sentences in the input file
            data, targets = get_batch(data_source, i, seq_len)
            _, targets_mask = get_batch(mask, i, seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets)

            probs = output_candidates_probs(output_flat, targets, targets_mask)

            hidden = repackage_hidden(hidden)

    return(probs)
    
def output_candidates_probs(output_flat, targets, mask):
    data_list = []
    log_probs = F.softmax(output_flat, dim=1)

    log_probs_np = log_probs.cpu().numpy()
    subset = mask.cpu().numpy().astype(bool)

    for scores, correct_label in zip(log_probs_np[subset], targets.cpu().numpy()[subset]):
        #print(idx2word[correct_label], scores[correct_label])
        data_list.append(-np.log2(scores))
    
    return(data_list)



# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    print("Loading the model")
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

eval_batch_size = 1
seq_len = 20

dictionary = dictionary_corpus.Dictionary(args.data)
vocab_size = len(dictionary)

input_path = get_directory(args.in_file)

out_file = args.out_dir + input_path[1] + '.out'

with open(args.in_file, 'r') as f:
    marked_sentences = f.readlines()

with open(out_file, 'w') as g:
    g.write('Sentence;TargetWord;Surprisal\n')
    for i in range(len(marked_sentences)):
        
        sentence = marked_sentences[i]
        
        sentence = sentence[:-1] + ' <eos>'
        split_sentence = sentence.split()
        clean_sentence = sentence.replace('*','')
        
        
        target_indices = [i for i,word in enumerate(split_sentence) if '*' in word]
        mask = np.zeros(np.shape(split_sentence))
        mask.put(target_indices,1)
        mask_data = batchify(torch.LongTensor(mask), eval_batch_size, args.cuda)
        
        current_sentence_file = input_path[0] + '/' + 'current_sentence'
        
        with open(current_sentence_file,'w') as f:
            f.write(clean_sentence)
        test_data = batchify(dictionary_corpus.tokenize(dictionary, current_sentence_file), eval_batch_size, args.cuda)
        
        
        outputs = evaluate(test_data, mask_data)
        
        target_words = [word.replace('*','') for i,word in enumerate(split_sentence) if '*' in word]
        
        try:
            target_dict_indices = [dictionary.word2idx[word] for word in target_words]
            
            for i in range(len(target_words)):
                final_sentence = clean_sentence[:-6]
                final_tw = target_words[i]
                final_surprisal = outputs[i][target_dict_indices[i]]
                final_output = '{0};{1};{2}\n'.format(final_sentence,final_tw,final_surprisal)
                g.write(final_output)
        except:
            print('One of the words in the following list is not in the dictionary of this model: "{0}"'
                  .format(', '.join(target_words)))


