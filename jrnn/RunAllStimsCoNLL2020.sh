#!/bin/bash

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/ainsworthdarnell_1998.stims\
							--output_file output/ainsworthdarnell_1998 \
							--record_type target \
							--ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/ito_2016.stims\
							--output_file output/ito_2016 \
							--record_type target \
							--ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/kim_2005.stims\
							--output_file output/kim_2005 \
							--record_type target \
							--ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/kutas_1993_sentences.stims\
							--output_file output/kutas_1993_sentences \
							--record_type target \
							--ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/osterhout_mobley_1995_exp2_controls.stims\
							--output_file output/osterhout_mobley_1995_exp2_controls \
							--record_type target \
							--ckpt 'data/ckpt-*'
							
bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/osterhout_mobley_1995_exp2_controls_wf.stims\
							--output_file output/osterhout_mobley_1995_exp2_controls_wf \
							--record_type target \
							--ckpt 'data/ckpt-*'
							
bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/osterhout_mobley_1995_exp2_pronouns.stims\
							--output_file output/osterhout_mobley_1995_exp2_pronouns \
							--record_type target \
							--ckpt 'data/ckpt-*'							

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/osterhout_mobley_1995_exp2_pronouns_wf.stims\
							--output_file output/osterhout_mobley_1995_exp2_pronouns_wf \
							--record_type target \
							--ckpt 'data/ckpt-*'	

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/urbach_2010_exp1.stims\
							--output_file output/urbach_2010_exp1 \
							--record_type target \
							--ckpt 'data/ckpt-*'	

bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/urbach_2010_exp2.stims\
							--output_file output/urbach_2010_exp2 \
							--record_type target \
							--ckpt 'data/ckpt-*'
							
bazel-bin/lm_1b/lm_1b_eval 	--mode run_n400_exp \
                           	--pbtxt data/graph-2016-09-10.pbtxt \
                           	--vocab_file data/vocab-2016-09-10.txt  \
                           	--input_data stimuli/urbach_2010_exp3.stims\
							--output_file output/urbach_2010_exp3 \
							--record_type target \
							--ckpt 'data/ckpt-*'

