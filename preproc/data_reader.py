import csv, os
import xml.etree.ElementTree as ET
from collections import defaultdict

from sklearn.model_selection import train_test_split

from constants import FNC, STANCE, NLI, TOPIC, LAPTOP, RESTAURANT, TARGET,\
    TOPIC_5WAY, STANCE_LABELS, FNC_LABELS, NLI_LABELS, TOPIC_LABELS, \
    TOPIC_5WAY_LABELS, ABSA_LABELS, TARGET_LABELS


def task2data_reader(task):
    if task == STANCE: # all data available
        return readSemEval2016Task6
    if task == FNC: # all data available
        return readFakeNewsChallengeData
    if task == NLI:  # test data not available - so we use every other dev example as test example
        return readMultinliData
    if task == TOPIC:  # all data available
        return readTopicBased
    if task == TOPIC_5WAY:
        return readTopic5Way
    if task == LAPTOP: # all data available
        return read_absa_laptops
    if task == RESTAURANT: # all data available
        return read_absa_restaurants
    if task == TARGET:  # all data available
        return read_target_dependent
    raise ValueError('No data reader available for %s.' % task)


def readSemEval2016Task6(datafolder="./data/", debug=True, num_instances=20):
    data_train = {"seq1": [], "seq2": [], "stance": [], "opinion_towards": [], "sentiment": [], "labels": []}
    data_dev = {"seq1": [], "seq2": [], "stance": [], "opinion_towards": [], "sentiment": [], "labels": []}
    data_test = {"seq1": [], "seq2": [], "stance": [], "opinion_towards": [], "sentiment": [], "labels": []}
    data_train, data_dev = parse_semeval_csv(os.path.join(datafolder, 'semeval2016-task6-stance/train.csv'), data_train, data_dev, "train", debug, num_instances)
    data_test, data_dev = parse_semeval_csv(os.path.join(datafolder, 'semeval2016-task6-stance/test.csv'), data_test, data_dev, "test", False, num_instances)  # setting debug to False to get all test instances

    # For the final task training, the dev set is used as part of the training set
    for i, inst in enumerate(data_dev["stance"]):
        data_train["seq1"].append(data_dev["seq1"][i])
        data_train["seq2"].append(data_dev["seq2"][i])
        data_train["stance"].append(data_dev["stance"][i])
        data_train["opinion_towards"].append(data_dev["opinion_towards"][i])
        data_train["sentiment"].append(data_dev["sentiment"][i])

    # sort the labels so that they are always in the same order so that we can
    # compute averaged positive and negative F1 (AGAINST, FAVOR, NONE)
    labels = sorted(list(set(data_train["stance"])))
    assert labels == STANCE_LABELS
    data_train["labels"] = labels
    data_dev["labels"] = labels
    data_test["labels"] = labels

    # we do not use the raw data ATM to correspond with the signature of the other data readers
    return data_train, data_dev, data_test


def parse_semeval_csv(filepath, empty_dict_1, empty_dict_2, mode, debug=False, num_instances=20):
    with open(filepath, 'r', encoding="latin-1") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = -1
        for row in csvreader:
            i += 1
            if i == 0:
                continue
            if debug and i >= num_instances+1:
                continue
            tweet, target, stance, opinion_towards, sentiment = row
            dict_chosen = empty_dict_1
            if target == "Hillary Clinton":
                dict_chosen = empty_dict_2
            if mode == "train" or target == "Hillary Clinton" or (mode == "test" and target == "Donald Trump"):
                dict_chosen["seq1"].append(target)
                dict_chosen["seq2"].append(tweet)
                dict_chosen["stance"].append(stance)
                dict_chosen["opinion_towards"].append(opinion_towards)
                dict_chosen["sentiment"].append(sentiment)
    return empty_dict_1, empty_dict_2


def readFakeNewsChallengeData(datafolder="./data/", debug=True, num_instances=20):
    data_train = {"seq1": [], "seq2": [], "stance": [], "labels": []}
    data_train = parseFakeNewsChallengeData(datafolder, "fakenewschallenge/train_bodies.csv", "fakenewschallenge/trainsplit_stances.csv", data_train, debug, num_instances)
    data_dev = {"seq1": [], "seq2": [], "stance": [], "labels": []}
    data_dev = parseFakeNewsChallengeData(datafolder, "fakenewschallenge/train_bodies.csv", "fakenewschallenge/devsplit_stances.csv", data_dev, debug, num_instances)
    data_test = {"seq1": [], "seq2": [], "stance": [], "labels": []}
    data_test = parseFakeNewsChallengeData(datafolder, "fakenewschallenge/competition_test_bodies.csv", "fakenewschallenge/competition_test_stances.csv", data_test, debug, num_instances)
    data_train["labels"] = sorted(data_train["labels"])
    assert data_train["labels"] == FNC_LABELS
    data_dev["labels"] = data_train["labels"]
    data_test["labels"] = data_train["labels"]
    return data_train, data_dev, data_test


def parseFakeNewsChallengeData(datafolder, datafile_bodies, datafile_stances, data_dict, debug, num_instances):
    id2body = {}
    with open(os.path.join(datafolder, datafile_bodies), 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = -1
        for row in csvreader:
            i += 1
            if i == 0:
                continue
            body_id, body = row
            id2body[body_id] = body

    with open(os.path.join(datafolder, datafile_stances), 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = -1
        for row in csvreader:
            i += 1
            if i == 0:
                continue
            if debug and i >= num_instances+1:
                continue
            headline, body_id, stance = row
            data_dict["seq1"].append(headline)
            data_dict["seq2"].append(id2body[body_id])
            data_dict["stance"].append(stance)

    for lab in set(data_dict["stance"]):
        data_dict["labels"].append(lab)

    return data_dict


def readMultinliData(datafolder="./data/", debug=True, num_instances=20):

    max_count = None
    if debug == True:
        max_count = num_instances+1

    data_train = {"seq1": [], "seq2": [], "stance": [], "genre": [], "labels": []}
    data_train, _ = parseMultinliFile(os.path.join(datafolder, 'multinli/multinli_0.9_train.txt'), data_train, {}, max_count, "train")
    data_dev = {"seq1": [], "seq2": [], "stance": [], "genre": [], "labels": []}
    data_test = {"seq1": [], "seq2": [], "stance": [], "genre": [], "labels": []}
    data_dev, data_test = parseMultinliFile(os.path.join(datafolder, 'multinli/multinli_0.9_dev_matched.txt'), data_dev, data_test, max_count, "test")

    return data_train, data_dev, data_test


def parseMultinliFile(filepath, data_1, data_2, max_count, mode):
    reading_dataset = open(filepath, "r", encoding='utf-8')
    # The script reads into those lists. If IDs for questions, supports or targets are defined, those are ignored.
    count, counti = 0, 0

    for line in reading_dataset:
        if max_count is None or count < max_count:
            lspl = line.strip("\n").split("\t")
            if len(lspl) == 15:
                gold_label, _, _, _, _, sentence1, sentence2, promptID, pairID, genre, _, _, _, _, _ = lspl
                if gold_label == "gold_label" or gold_label == "-":
                    continue
                data_dict = data_1
                if mode == "train" or (mode == "test" and count % 2 == 0):
                    data_dict = data_1
                elif mode == "test":
                    data_dict = data_2
                data_dict["seq1"].append(sentence1)
                data_dict["seq2"].append(sentence2)
                data_dict["stance"].append(gold_label)
                data_dict["genre"].append(genre)
                count += 1

    for lab in set(data_1["stance"]):
        data_1["labels"].append(lab)
    data_1["labels"] = sorted(data_1["labels"])
    assert data_1["labels"] == NLI_LABELS

    if data_2 != {}:
        for lab in set(data_2["stance"]):
            data_2["labels"].append(lab)
        data_2["labels"] = sorted(data_2["labels"])
        assert data_2["labels"] == NLI_LABELS

    return data_1, data_2


def readTopicBased(datafolder="./data/", debug=True, num_instances=20):
    topic_based_path = os.path.join(datafolder, 'semeval2016-task4b-topic-based-sentiment')
    train_path = os.path.join(topic_based_path, '100_topics_XXX_tweets.topic-two-point.subtask-BD.train.gold_downloaded.tsv')
    dev1_path = os.path.join(topic_based_path, '100_topics_XXX_tweets.topic-two-point.subtask-BD.dev.gold_downloaded.tsv')
    dev2_path = os.path.join(topic_based_path, '100_topics_XXX_tweets.topic-two-point.subtask-BD.devtest.gold_downloaded.tsv')
    test_data_path = os.path.join(topic_based_path, 'SemEval2016-task4-test.subtask-BD.txt')
    test_labels_path = os.path.join(topic_based_path, 'SemEval2016_task4_subtaskB_test_gold.txt')

    for path_ in [topic_based_path, train_path, dev1_path, dev2_path, test_data_path, test_labels_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_topic_based(train_path, debug, num_instances)
    data_dev1 = parse_topic_based(dev1_path, debug, num_instances)
    data_dev2 = parse_topic_based(dev2_path, debug, num_instances)
    data_test = parse_topic_test_data(test_data_path, test_labels_path)
    assert data_train["labels"] == TOPIC_LABELS
    data_dev1["labels"] = data_train["labels"]
    data_test["labels"] = data_train["labels"]

    # add the second dev data to the train set
    data_train["seq1"] += data_dev2["seq1"]
    data_train["seq2"] += data_dev2["seq2"]
    data_train["stance"] += data_dev2["stance"]
    return data_train, data_dev1, data_test


def readTopic5Way(datafolder="./data/", debug=True, num_instances=20):
    topic_based_path = os.path.join(datafolder, 'semeval2016-task4c-topic-based-sentiment')
    train_path = os.path.join(topic_based_path, '100_topics_100_tweets.topic-five-point.subtask-CE.train.gold_downloaded.tsv')
    dev1_path = os.path.join(topic_based_path, '100_topics_100_tweets.topic-five-point.subtask-CE.dev.gold_downloaded.tsv')
    dev2_path = os.path.join(topic_based_path, '100_topics_100_tweets.topic-five-point.subtask-CE.devtest.gold_downloaded.tsv')
    test_data_path = os.path.join(topic_based_path, 'SemEval2016-task4-test.subtask-CE.txt')
    test_labels_path = os.path.join(topic_based_path, 'SemEval2016_task4_subtaskC_test_gold.txt')

    for path_ in [topic_based_path, train_path, dev1_path, dev2_path,
                  test_data_path, test_labels_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_topic_based(train_path, debug, num_instances)
    data_dev1 = parse_topic_based(dev1_path, debug, num_instances)
    data_dev2 = parse_topic_based(dev2_path, debug, num_instances)
    data_test = parse_topic_test_data(test_data_path, test_labels_path)
    assert data_train["labels"] == TOPIC_5WAY_LABELS
    data_dev1["labels"] = data_train["labels"]
    data_test["labels"] = data_train["labels"]

    # add the second dev data to the train set
    data_train["seq1"] += data_dev2["seq1"]
    data_train["seq2"] += data_dev2["seq2"]
    data_train["stance"] += data_dev2["stance"]
    return data_train, data_dev1, data_test


def parse_topic_based(file_path, debug=False, num_instances=20):
    data = {"seq1": [], "seq2": [], "stance": []}
    with open(file_path) as f:
        for i, line in enumerate(f):
            id_, target, sentiment, tweet = line.split('\t')
            try:
                sentiment = float(sentiment)
            except ValueError:
                pass
            if debug and i >= num_instances+1:
                continue
            if tweet.strip() == 'Not Available':
                continue
            data["seq1"].append(target)
            data["seq2"].append(tweet)
            data["stance"].append(sentiment)

    # we have to sort the labels so that they're in the order
    # -2,-1,0,1,2 and are mapped to 0,1,2,3,4 (for subtask C)
    data["labels"] = sorted(list(set(data["stance"])))
    return data


def parse_topic_test_data(examples_path, labels_path):
    # Note: no debugging for the test data (20k tweets for subtask C)
    data = {"seq1": [], "seq2": [], "stance": []}
    with open(examples_path) as f_examples, open(labels_path) as f_labels:
        for i, (line_examples, line_labels) in enumerate(zip(f_examples, f_labels)):
            _, examples_target, _, *tweet = line_examples.strip().split('\t')
            # two lines contain a tweet, for some reason
            _, labels_target, sentiment, *_ = line_labels.strip().split('\t')
            # one test tweet contains a tab character
            if isinstance(tweet, list):
                tweet = '\t'.join(tweet)
            try:
                sentiment = float(sentiment)
            except ValueError:
                pass

            assert examples_target == labels_target,\
                '%s != %s at line %d in files %s and %s.' % (
                examples_target, labels_target, i, examples_path, labels_path)

            if tweet.strip() == 'Not Available':
                continue
            data["seq1"].append(examples_target)
            data["seq2"].append(tweet)
            data["stance"].append(sentiment)
    data["labels"] = sorted(list(set(data["stance"])))
    return data


def read_absa_laptops(datafolder="./data/", debug=True, num_instances=20):
    return read_absa('laptops', datafolder, debug, num_instances)


def read_absa_restaurants(datafolder="./data/", debug=True, num_instances=20):
    return read_absa('restaurants', datafolder, debug, num_instances)


def read_absa(domain, datafolder="./data/", debug=True, num_instances=20):
    assert domain in ['laptops', 'restaurants'], '%s is not a valid domain.' % domain
    absa_path = os.path.join(datafolder, 'semeval2016-task5-absa-english')
    train_path = os.path.join(absa_path, '%s_english_training.xml' % domain)
    test_path = os.path.join(absa_path, '%s_english_test.xml' % domain)
    for path_ in [absa_path, train_path, test_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_absa(train_path, debug, num_instances)
    data_test = parse_absa(test_path)

    # trial data is a subset of training data; instead we split the train data
    data_train, data_dev = split_train_data(data_train)
    return data_train, data_dev, data_test


def parse_absa(file_path, debug=False, num_instances=20):
    """
    Extracts all reviews from an XML file and returns them as a list of Review objects.
    Adds a NONE aspect to all sentences with no aspect.
    :param file_path: the path of the XML file
    :return: a list of Review objects each containing a list of Sentence objects and other attributes
    """
    data = {"seq1": [], "seq2": [], "stance": []}
    e = ET.parse(file_path).getroot()
    for i, review_e in enumerate(e):
        if debug and i >= num_instances+1:
            continue
        for sentence_e in review_e.find('sentences'):
            text = sentence_e.find('text').text
            # we do not care about sentences that do not contain an aspect
            if sentence_e.find('Opinions') is not None:
                for op in sentence_e.find('Opinions'):
                    # the category is of the form ENTITY#ATTRIBUTE, e.g. LAPTOP#GENERAL
                    target = ' '.join(op.get('category').split('#'))
                    polarity = op.get('polarity')
                    data['seq1'].append(target)
                    data['seq2'].append(text)
                    data['stance'].append(polarity)
    data["labels"] = sorted(list(set(data["stance"])))
    assert data["labels"] == ABSA_LABELS
    return data


def read_target_dependent(datafolder="./data/", debug=True, num_instances=20):
    target_dependent_path = os.path.join(datafolder, 'target-dependent')
    train_path = os.path.join(target_dependent_path, 'train.raw')
    test_path = os.path.join(target_dependent_path, 'test.raw')
    for path_ in [target_dependent_path, train_path, test_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_target_dependent(train_path, debug, num_instances)
    data_test = parse_target_dependent(test_path)
    data_train, data_dev = split_train_data(data_train)
    return data_train, data_dev, data_test


def parse_target_dependent(file_path, debug=False, num_instances=20):
    data = {"seq1": [], "seq2": [], "stance": []}
    with open(file_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 3 == 0:  # the tweet is always first
                data["seq2"].append(line.strip())
            elif i % 3 == 1:  # followed by the target
                data["seq1"].append(line.strip())
            elif i % 3 == 2:  # followed by the sentiment
                data["stance"].append(line.strip())
            if debug and i >= num_instances+1:
                continue
        assert len(data["seq1"]) == len(data["seq2"]) == len(data["stance"]),\
            'Error: %d != %d != %d.' % (len(data["seq1"]), len(data["seq2"]),
                                        len(data["stance"]))

    # replace the placeholder $T$ in every tweet with the target
    for i in range(len(data["seq1"])):
        target = data["seq1"][i]
        data["seq2"][i] = data["seq2"][i].replace("$T$", target)
    data["labels"] = sorted(list(set(data["stance"])))
    assert data["labels"] == TARGET_LABELS
    return data


def split_train_data(data_train):
    """Split the train data into train and dev data."""
    train_ids, _ = train_test_split(range(len(data_train['seq1'])),
                                    test_size=0.1, random_state=42)
    data_dev = defaultdict(list)
    new_data_train = defaultdict(list)
    for key, examples in data_train.items():
        if key == 'labels':
            continue
        # no numpy indexing, so we iterate over the examples
        for i, example in enumerate(examples):
            if i in train_ids:
                new_data_train[key].append(example)
            else:
                data_dev[key].append(example)
    new_data_train['labels'] = data_train['labels']
    data_dev['labels'] = data_train['labels']
    return new_data_train, data_dev


if __name__ == "__main__":
    readMultinliData(datafolder="../data/")
