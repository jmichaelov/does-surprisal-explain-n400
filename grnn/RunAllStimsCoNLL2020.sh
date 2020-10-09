#!/bin/bash

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/osterhout_mobley_1995_exp2_controls.stims -o data/current_experiment/output/ 

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/osterhout_mobley_1995_exp2_pronouns.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/osterhout_mobley_1995_exp2_controls_wf.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/osterhout_mobley_1995_exp2_pronouns_wf.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/ainsworthdarnell_1998.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/ito_2016.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/kim_2005.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/kutas_1993_sentences.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/osterhout_1992_exp2.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/urbach_2010_exp1.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/urbach_2010_exp2.stims -o data/current_experiment/output/

python src/language_models/run_conll2020_experiment.py --data data/lm/English/ --checkpoint data/lm/English/hidden650_batch128_dropout0.2_lr20.0.pt --cuda -i data/current_experiment/stimuli/urbach_2010_exp3.stims -o data/current_experiment/output/
