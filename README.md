# mtl-disparate
Code for NAACL 2018 paper ["Multi-task Learning of Pairwise Sequence Classification Tasks Over Disparate Label Spaces"](https://arxiv.org/abs/1802.09913) by Isabelle Augenstein, Sebastian Ruder, Anders Søgaard

Note that this is research code and will not be maintained to e.g. ensure compatibility with more recent library versions.


Requirements:

- Tensorflow 1.5
- Numpy 1.12.1
- sklearn 0.18.1
- scipy

Steps to run:

- run data/download_data.sh to download and extract data
- preproc/data_reader.py tests if all the data readers work
- preproc/fnc_data_splits.py to split the FNC training dataset into a training and dev set
- main.py trains models

# Datasets

## SemEval 2016 Task 6 Stance detection

- [Task website](http://alt.qcri.org/semeval2016/task6/)

## Fake News Challenge (FNC)

- [Task website](http://www.fakenewschallenge.org/)

## Multi-NLI

- [Task website](http://www.nyu.edu/projects/bowman/multinli/)

## SemEval 2016 Task 4 Subtask B Topic-based Twitter sentiment analysis

- [Task website](http://alt.qcri.org/semeval2016/task4/)
- [Task description paper](https://aclweb.org/anthology/S/S16/S16-1001.pdf)
- Note: Same dataset was used as rerun in [2017](http://alt.qcri.org/semeval2017/task4/)

## SemEval 2016 Task 5 Subtask 1 Slot 3 Aspect-based sentiment analysis

- [Task website](http://alt.qcri.org/semeval2016/task5/)

## Clickbait Challenge 2017

- [Task website](http://www.clickbait-challenge.org/)