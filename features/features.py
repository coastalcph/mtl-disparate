"""
Methods to create the similarity and diversity features used in
Ruder & Plank (2017).
"""

import sys
import os
import numpy as np
np.seterr(all='raise')
import scipy.stats
import scipy.spatial

from preproc.map import tokenize, lower, deep_map, deep_seq_map
from constants import SIMILARITY_FEATURES, DIVERSITY_FEATURES, SIM, DIV

# ------------ Feature methods ------------


def create_features(feature_sets, task2examples, vocab, save_path):
    """
    Retrieve the feature representations of a list of examples.
    :param feature_sets: a list containing the names of features to be used
    :param task2examples: mapping of tasks to lists of untokenized texts
    :param vocab: the Vocabulary object
    :param save_path: the directory where the features should be stored
    :return: a mapping of tasks to feature representations of shape
            (num_examples, num_features); the features correspond to the order
            of the data; first training examples, then dev, then test
    """
    # create the features for each example in each task
    task2features = {t: [] for t in task2examples.keys()}

    # get the feature names
    feature_names = []
    if SIM in feature_sets:
        feature_names += SIMILARITY_FEATURES
    if DIV in feature_sets:
        feature_names += DIVERSITY_FEATURES

    print("Trying to find feature files in", save_path)
    if os.path.exists(save_path) and os.path.isdir(save_path) and len(
            os.listdir(save_path)) > 0:
        feature_dim = None
        for task in task2examples.keys():
            assert task in os.listdir(save_path),\
                'Error: No saved features available for task %s in dir %s.' \
                % (task, save_path)
        print("Task2features")
        print(task2features)
        print("Files in features folder")
        print(os.listdir(save_path))
        for task in os.listdir(save_path):
            # then we don't need to load it
            if not task in task2features.keys():
                continue
            with open(os.path.join(save_path, task), 'r') as f:
                for line in f:
                    features = np.fromstring(line.strip('[]'), dtype=float,
                                             sep=' ')
                    if feature_dim is None:
                        feature_dim = len(features)
                    assert feature_dim == len(feature_names),\
                        'Error: # of loaded features %d != # of specified '\
                        'features %d.' % (feature_dim, len(feature_names))
                    assert feature_dim == len(features),\
                        'Error: Different # of features among examples, ' \
                        'i.e. %d and %d.' % (feature_dim, len(features))
                    task2features[task].append(features)
            print('Loaded %d-d features for %s from %s...'
                  % (feature_dim, task, save_path))
        return task2features

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # tokenize and lower-case the documents
    for task, examples in task2examples.items():
        examples = deep_map(examples, tokenize)
        examples = deep_seq_map(examples, lower)
        task2examples[task] = examples

    # get the term distribution of the data for each task (shape (vocab_size,) )
    # and for each example (shape (num_examples, vocab_size) )
    task2task_term_dist = {}
    for task, examples in task2examples.items():
        task2task_term_dist[task] = get_term_dist(examples, vocab.sym2id)

    for task, examples in task2examples.items():
        for i, example in enumerate(examples):
            term_dist = get_term_dist([example], vocab.sym2id)
            features = []
            for f_name in feature_names:
                # check whether feature belongs to similarity-based features,
                # diversity-based features, etc.
                if f_name in SIMILARITY_FEATURES:
                    # compute the similarity with regard to each task
                    for target_task in task2examples.keys():
                        f = similarity_name2value(
                            f_name, term_dist, task2task_term_dist[target_task])
                        if np.isnan(f).any() or np.isinf(f).any():
                            if type(f) != list:
                                f = [0 for ff in f]
                            elif type(f) == int:
                                f = 0
                            elif type(f) == float:
                                f = 0.0
                        features.append(f)
                elif f_name in DIVERSITY_FEATURES:
                    f = diversity_feature_name2value(
                        f_name, example, task2task_term_dist[task],
                        vocab.sym2id)
                    if np.isnan(f).any() or np.isinf(f).any():
                        if type(f) != list:
                            f = [0 for ff in f]
                        elif type(f) == int:
                            f = 0
                        elif type(f) == float:
                            f = 0.0
                    features.append(f)
                else:
                    raise ValueError('%s is not a valid feature name.' % f_name)
            #assert not np.isnan(features).any(), 'Error: NAN value in array.'
            #assert not np.isinf(features).any(), 'Error: inf or -inf value.'
            task2features[task].append(features)
            if i % 100 == 0 and i > 0:
                print('%s. Created features for %d examples.' % (task, i))
        task2features[task] = np.array(task2features[task])

    # z-normalize the feature scores
    feature_values = scipy.stats.zscore(np.vstack([f for f in
                                        task2features.values()]), axis=0)
    start_idx = 0
    for task, features in task2features.items():
        task2features[task] = feature_values[
                              start_idx:start_idx+features.shape[0], :]
        start_idx += features.shape[0]

        # write the features to the corresponding file
        file_path = os.path.join(save_path, task)
        with open(file_path, 'w') as f:
            for example_features in task2features[task]:
                # set max_line_width so that features don't wrap across lines
                f.write('%s\n' % np.array_str(example_features,
                                              max_line_width=sys.maxsize))
        print('Wrote %s %d-d features to %s...' % (task, len(feature_names),
                                                   file_path))
    print('Created features.')
    return task2features


def get_term_dist(docs, word2id, lowercase=True):
    """
    Calculates the term distribution of a list of documents.
    :param docs: a list of tokenized docs; can also contain a single document
    :param vocab: the Vocabulary object
    :param lowercase: lower-case the input data
    :return: the term distribution of the input documents,
             i.e. a numpy array of shape (vocab_size,)
    """
    term_dist = np.zeros(len(word2id))
    for doc in docs:
        for word in doc:
            if lowercase:
                word = word.lower()
            if word in word2id:
                term_dist[word2id[word]] += 1

    # normalize absolute freqs to obtain a relative frequency term distribution
    term_dist /= np.sum(term_dist)
    if np.isnan(np.sum(term_dist)):
        # the sum is nan if docs only contains one document and that document
        # has no words in the vocabulary
        term_dist = np.zeros(len(word2id))
    return term_dist


# ------------ Similarity features ------------

def jensen_shannon_divergence(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr1, avg_repr))
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim


def renyi_divergence(repr1, repr2, alpha=0.99):
    """Calculates Renyi divergence (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R.C3.A9nyi_divergence)."""
    try:
        log_sum = np.sum([np.power(p, alpha) / np.power(q, alpha-1) for (p, q) in zip(repr1, repr2)])
        sim = 1 / (alpha - 1) * np.log(log_sum)
    except FloatingPointError: # division by 0 error
        return 0
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim


def cosine_similarity(repr1, repr2):
    """Calculates cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)."""
    if repr1 is None or repr2 is None:
        return 0
    assert not (np.isnan(repr2).any() or np.isinf(repr2).any())
    assert not (np.isnan(repr1).any() or np.isinf(repr1).any())
    sim = 1 - scipy.spatial.distance.cosine(repr1, repr2)
    if np.isnan(sim):
        # the similarity is nan if no term in the document is in the vocabulary
        return 0
    return sim


def euclidean_distance(repr1, repr2):
    """Calculates Euclidean distance (https://en.wikipedia.org/wiki/Euclidean_distance)."""
    sim = np.sqrt(np.sum([np.power(p-q, 2) for (p, q) in zip(repr1, repr2)]))
    return sim


def variational_distance(repr1, repr2):
    """Also known as L1 or Manhattan distance (https://en.wikipedia.org/wiki/Taxicab_geometry)."""
    sim = np.sum([np.abs(p-q) for (p, q) in zip(repr1, repr2)])
    return sim


def kl_divergence(repr1, repr2):
    """Calculates Kullback-Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)."""
    sim = scipy.stats.entropy(repr1, repr2)
    return sim


def bhattacharyya_distance(repr1, repr2):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    try:
        sim = - np.log(np.sum([np.sqrt(p*q) for (p, q) in zip(repr1, repr2)]))
    except FloatingPointError:  # division by 0 error
        return 0
    assert not np.isnan(sim), 'Error: Similarity is nan.'
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim


def similarity_name2value(s_name, repr1, repr2):
    """Given a similarity function name, return the corresponding similarity function value."""
    if s_name == 'jensen-shannon':
        return jensen_shannon_divergence(repr1, repr2)
    if s_name == 'renyi':
        return renyi_divergence(repr1, repr2)
    if s_name == 'cos' or s_name == 'cosine':
        return cosine_similarity(repr1, repr2)
    if s_name == 'euclidean':
        return euclidean_distance(repr1, repr2)
    if s_name == 'variational':
        return variational_distance(repr1, repr2)
    if s_name == 'kl':
        return kl_divergence(repr1, repr2)
    if s_name == 'bhattacharyya':
        return bhattacharyya_distance(repr1, repr2)
    raise ValueError('%s is not a valid feature name.' % s_name)


# ------------ Diversity features ------------

def number_of_word_types(example):
    """Counts the number of word types of the example."""
    return len(set(example))


def type_token_ratio(example):
    """Calculates the type-token ratio of the example."""
    return number_of_word_types(example) / len(example)


def entropy(example, train_term_dist, word2id):
    """Calculates Entropy (https://en.wikipedia.org/wiki/Entropy_(information_theory))."""
    summed = 0
    for word in set(example):
        if word in word2id:
            p_word = train_term_dist[word2id[word]]
            summed += p_word * np.log(p_word)
    return - summed


def simpsons_index(example, train_term_dist, word2id):
    """Calculates Simpson's Index (https://en.wikipedia.org/wiki/Diversity_index#Simpson_index)."""
    score = np.sum([np.power(train_term_dist[word2id[word]], 2) if word in word2id else 0
                    for word in set(example)])
    return score


def quadratic_entropy(example, train_term_dist, word2id, word2vec):
    """Calculates Quadratic Entropy."""
    assert word2vec is not None, ('Error: Word vector representations have to '
                                  'be available for quadratic entropy.')
    summed = 0
    for word_1 in set(example):
        if word_1 not in word2id or word_1 not in word2vec:
            continue  # continue as the product will be 0
        for word_2 in set(example):
            if word_2 not in word2id or word_2 not in word2vec:
                continue  # continue as the product will be 0
            p_1 = train_term_dist[word2id[word_1]]
            p_2 = train_term_dist[word2id[word_2]]
            vec_1 = word2vec[word_1]
            vec_2 = word2vec[word_2]
            sim = cosine_similarity(vec_1, vec_2)
            summed += sim * p_1 * p_2
    return summed


def renyi_entropy(example, domain_term_dist, word2id):
    """Calculates Rényi Entropy (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy)."""
    alpha = 0.99
    summed = np.sum([np.power(domain_term_dist[word2id[word]], alpha) if word in word2id else 0 for word in set(example)])
    if summed == 0:
        # 0 if none of the words appear in the dictionary;
        # set to a small constant == low prob instead
        summed = 0.0001
    score = 1 / (1 - alpha) * np.log(summed)
    return score


def diversity_feature_name2value(f_name, example, task_term_dist, word2id):
    """
    Given a feature name, return the corresponding feature value.
    :param f_name: the name of the feature
    :param example: the tokenised example document
    :param task_term_dist: the term distribution of the task of the example
    :param word2id: the word-to-id mapping
    :param word2vec: a mapping of a word to its word vector representation (e.g. GloVe or word2vec)
    :return: the value of the corresponding feature
    """
    if f_name == 'num_word_types':
        return number_of_word_types(example)
    if f_name == 'type_token_ratio':
        return type_token_ratio(example)
    if f_name == 'entropy':
        return entropy(example, task_term_dist, word2id)
    if f_name == 'simpsons_index':
        return simpsons_index(example, task_term_dist, word2id)
    # if f_name == 'quadratic_entropy':
    #     return quadratic_entropy(example, train_term_dist, word2id, word2vec)
    if f_name == 'renyi_entropy':
        return renyi_entropy(example, task_term_dist, word2id)
    raise ValueError('%s is not a valid feature name.' % f_name)
