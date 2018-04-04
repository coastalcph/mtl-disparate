#!/usr/bin/env bash

# Download the SemEval 2016 Task 6 Stance detection dataset
mkdir semeval2016-task6-stance ; cd semeval2016-task6-stance
wget http://alt.qcri.org/semeval2016/task6/data/uploads/stancedataset.zip
wget http://alt.qcri.org/semeval2016/task6/data/uploads/semeval2016-task6-trialdata.txt
curl -L "https://drive.google.com/uc?export=download&id=0B2Z1kbILu3YtenFDUzM5dGZEX2s" > downloaded_Donald_Trump.txt
unzip stancedataset.zip -d . ; mv StanceDataset/* .
rm stancedataset.zip ; rm -r StanceDataset __MACOSX
cd ..

# Download the Fake News Challenge datset
mkdir fakenewschallenge ; cd fakenewschallenge
wget https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_stances.csv
wget https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_bodies.csv
wget https://github.com/FakeNewsChallenge/fnc-1/archive/master.zip
unzip master.zip -d . ; mv fnc-1-master/* .
rm -r fnc-1-master ; rm master.zip
cd ..

# Download the Multi-NLI dataset
mkdir multinli ; cd multinli
wget http://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip
unzip multinli_0.9.zip -d . ; mv multinli_0.9/* .
rm multinli_0.9.zip ; rm -r multinli_0.9
cd ..

# Download the SemEval 2016 Task 4 Subtask B Topic-based Twitter sentiment analysis dataset
mkdir semeval2016-task4b-topic-based-sentiment ; cd semeval2016-task4b-topic-based-sentiment
curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtSGpKcjQ3cnhldmc" > semeval2016_task4b_topic-based_sentiment.zip
unzip semeval2016_task4b_topic-based_sentiment.zip -d .
rm semeval2016_task4b_topic-based_sentiment.zip
cd ..

# Download the SemEval 2016 Task 4 Subtask C Topic-based 5-way Twitter sentiment analysis dataset
mkdir semeval2016-task4c-topic-based-sentiment ; cd semeval2016-task4c-topic-based-sentiment
curl -L "https://drive.google.com/uc?export=download&id=1eS67x5vedrzVVk-tcyKSrumigbJKuqH-" > semeval2016_task4c_topic-based_sentiment.zip
unzip semeval2016_task4c_topic-based_sentiment.zip -d .
rm semeval2016_task4c_topic-based_sentiment.zip
cd ..

# Download the SemEval 2016 Task 5 Aspect-based sentiment analysis dataset
mkdir semeval2016-task5-absa-english ; cd semeval2016-task5-absa-english
curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtbTJnUHRIdFBULTg" > semeval2016_task5_absa_english.zip
unzip semeval2016_task5_absa_english.zip -d .
rm semeval2016_task5_absa_english.zip
cd ..

# Download the target-dependent sentiment analysis dataset of Dong et al. (2014):
# Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification
mkdir target-dependent ; cd target-dependent
curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtTW1SZjItWFlxUUU" > target_dependent.zip
unzip target_dependent.zip -d .
rm target_dependent.zip
cd ..
