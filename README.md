# How well does surprisal explain N400 amplitude under different experimental conditions?
This repository contains the code for 'How well does surprisal explain N400 amplitude under different experimental conditions? (Michaelov and Bergen, 2020)

## How to run the experiments presented in the paper
### Requirements
* The Python code, in its current version, is written to run in `Python 3.6.10`, and requires up-to-date versions of `tensorflow`, `pytorch`, and their dependencies. 
* The JRNN requires `Bazel` to run.
* The R code for statistical analysis is written to run in `R 3.6.3`, and requires `tidyverse`,`lme4`, and `lmerTest`.

### JRNN experiments
1. Run `download_files.sh` to download all the necessary data files.
2. Run `build_bazel.sh` to build the model in Bazel (required).
3. Run `RunAllStimsCoNLL2020.sh` to run the experiments for all stimuli.

### GRNN experiments
1. Run `download_files.sh` to download all the necessary data files.
3. Run `RunAllStimsCoNLL2020.sh` to run the experiments for all stimuli.

### Statistical Analyses
1. The code chunks in `CoNLL2020Submission.Rmd` can be run individually, or the file can be 'knitted' into an html file (`CoNLL2020Submission.html`).

## Sources

### RNN-LM Models and Code
The code in this repository is based on the code released from two papers:

* Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N., & Wu, Y. (2016). [Exploring the limits of language modeling](https://arxiv.org/abs/1602.02410). *arXiv preprint arXiv:1602.02410*. 
	* Code and pretrained models previously available [here](https://github.com/tensorflow/models/tree/master/research/lm_1b) under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
	* The pretrained model provided in this release is the JRNN.

* Gulordava, K., Bojanowski, P., Grave, É., Linzen, T., & Baroni, M. (2018, June). [Colorless Green Recurrent Networks Dream Hierarchically](https://www.aclweb.org/anthology/N18-1108/). In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)* (pp. 1195-1205).
	* Code and pretrained models available [here](https://github.com/facebookresearch/colorlessgreenRNNs) under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).
	* The pretrained model provided in this release is the GRNN.


### Experimental Stimuli
The experimental stimuli used in this research all appear in the appendices or supplementary material of the relevant papers:

* Urbach, T. P., & Kutas, M. (2010). [Quantifiers more or less quantify on-line: ERP evidence for partial incremental interpretation](https://doi.org/10.1016/j.jml.2010.03.008). *Journal of Memory and Language, 63(2)*, 158-179. (Stimuli available in appendix)

* Kutas, M. (1993). In the company of other words: [Electrophysiological evidence for single-word and sentence context effects](https://doi.org/10.1080/01690969308407587). *Language and Cognitive Processes, 8(4)*, 533-572. (Stimuli available in appendix)

* Ito, A., Corley, M., Pickering, M. J., Martin, A. E., & Nieuwland, M. S. (2016). [Predicting form and meaning: Evidence from brain potentials](https://doi.org/10.1016/j.jml.2015.10.007). *Journal of Memory and Language, 86*, 157-171. (Stimuli available in supplementary materials)

* Osterhout, L., & Mobley, L. A. (1995). [Event-related brain potentials elicited by failure to agree](https://doi.org/10.1006/jmla.1995.1033). *Journal of Memory and language, 34(6)*, 739-773. (Stimuli available in appendix)

* Ainsworth-Darnell, K., Shulman, H. G., & Boland, J. E. (1998). [Dissociating brain responses to syntactic and semantic anomalies: Evidence from event-related potentials](https://doi.org/10.1006/jmla.1997.2537). *Journal of Memory and Language, 38(1)*, 112-130. (Stimuli available in appendix)

* Kim, A., & Osterhout, L. (2005). [The independence of combinatory semantic processing: Evidence from event-related potentials](https://doi.org/10.1016/j.jml.2004.10.002). *Journal of Memory and Language, 52(2)*, 205-225. (Stimuli available in appendix)


## Contents and changes to files
### JRNN
The `jrnn` folder contains all the files for the JRNN model.
* Unless stated otherwise, all files from the original release are unchanged.
* The `lm_1b` folder contains the Python code that runs the experiments.
	* `lm_1b_eval.py` is the file that runs each experiment. This has been  substantially changed from the original release.
* The `output` folder contains the output of the experiment.
* The `stimuli` folder contains the stimuli that were used to run this experiment.
* `download_files.sh` has been added, and downloads all the files necessary to run the model.
* `build_bazel.sh` has been added, and builds the model in Bazel (as described in `README.md`).
* `RunAllStimsCoNLL2020.sh` has been added, and runs the experiments for all stimuli.
* `WORKSPACE` is a file required by Bazel.
* For completeness, the `LICENSE` file that has been used for the `tensorflow/models` repository since 2016 (see [repository](https://github.com/tensorflow/models/)) has also been added.

### GRNN
The `grnn` folder contains all the files for the GRNN model.
* Unless stated otherwise, all files from the original release are unchanged, though some have been removed to save space.
* `RunAllStimsCoNLL2020.sh` has been added, and runs the experiments for all stimuli.
* `download_files.sh` has been added, and downloads all the files necessary to run the model.
* The `data` folder contains the stimuli used to run the experiments.
* The `src` folder contains code used to run experiments.
	* `src/language_models/run_conll2020_experiment.py` has been added, and is the python code for running each individual experiment. It is partially based on `evaluate_target_word.py` (by the original authors).

### Stats
The `stats` folder contains an RMarkdown file that shows the statistical analyses run on the outputs of the models and the resulting html file.


## License(s)

The two models used and the code associated with them (including new files created by the authors to run experiments) are licensed under the [original copyright licenses associated with these models](#RNN-LM-Models-and-Code). All other code is licensed under the MIT License.