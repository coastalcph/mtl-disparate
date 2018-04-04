"""
Utility methods for logging and analyzing results.
"""

from collections import defaultdict
from datetime import datetime
import numpy as np

from sklearn.metrics import recall_score, mean_absolute_error, f1_score,\
    accuracy_score

from constants import *
import os


FORMAT = '%Y-%m-%d-%H%M%S'

RECALL = 'recall'
MAE = 'mae'
TOPIC_BASED_SCORES = [RECALL, MAE]

def stance_postproc_init(vocab):
    inds = []
    for id, tok in vocab.id2sym.items():
        if "trump" in tok or "donald" in tok:
            inds.append(id)
    return inds

def postproc_stance(inds, placeholders, batch, p):
    for i, b in enumerate(batch[placeholders["seq1"]]):
        in_ind = False
        for ind in inds:
            if ind in batch[placeholders["seq1"]][i]:
                in_ind = True
                break
        # labels are always: AGAINST, FAVOR, NONE
        if in_ind:
            if p[i][0] > p[i][1]:
                p[i][0] = 1.0
            else:
                p[i][1] = 1.0
    return p


def task2score(task, y_true, y_pred, topics):
    if task == STANCE:
        return macro_averaged_pos_neg_f1_score(y_true, y_pred)
    if task == TOPIC:
        return topic_based_macro_averaged_score(y_true, y_pred, topics, RECALL)
    if task == TOPIC_5WAY:
        return topic_based_macro_averaged_score(y_true, y_pred, topics, MAE)
    if task in [LAPTOP, RESTAURANT]:
        return accuracy_score(y_true, y_pred)
    if task in [TARGET]:
        return f1_score(y_true, y_pred, average='macro')
    return f1_score(y_true, y_pred, average='micro')


def macro_averaged_pos_neg_f1_score(y_true, y_pred):
    """Compute the macro-average of the favor and against F1 scores for stance
    detection."""
    # order of labels is AGAINST, FAVOR, NONE
    f1_scores = f1_score(y_true, y_pred, average=None)
    return np.mean([f1_scores[0], f1_scores[1]])


def topic_based_macro_averaged_score(y_true, y_pred, topics, score):
    """
    Compute score macro-averaged across topics. Score is macro-averaged recall
    for subtask B and mean absolute error for subtask C.
    """
    assert score in TOPIC_BASED_SCORES, 'Error: %s is not valid.' % score
    scores = []
    topic2y_true = defaultdict(list)
    topic2y_pred = defaultdict(list)

    # aggregate the labels and predictions for each topic
    for y_t, y_p, topic in zip(y_true, y_pred, topics):
        topic2y_true[str(topic)].append(y_t)
        topic2y_pred[str(topic)].append(y_p)

    for topic in topic2y_true.keys():
        y_true_topic = topic2y_true[str(topic)]
        y_pred_topic = topic2y_pred[str(topic)]
        if score == RECALL:
            score_value = recall_score(y_true_topic, y_pred_topic, average='macro')
        else:
            # for MAE, we have to manually perform macro-averaging
            # labels are 0-4 and correspond to original labels -2,-1,0,1,2
            temp_scores = []
            for label_id in range(5):
                true_pred_pairs = [(y_t, y_p) for y_t, y_p in
                                   zip(y_true_topic, y_pred_topic)
                                   if y_t == label_id]
                if len(true_pred_pairs) == 0:
                    # some topics do not appear with a certain label
                    continue
                y_true_temp, y_pred_temp = zip(*true_pred_pairs)
                temp_scores.append(mean_absolute_error(y_true_temp, y_pred_temp))
            score_value = np.mean(temp_scores)
        scores.append(score_value)
    return np.mean(scores)


def log_results(options, task_score, f1_score, relabel_score, task):
    """
    Log the results to a file.
    :param options: the options used as input to the script
    :param task_score: the task-specific score achieved on the test set
    :param f1_score: the micro-averaged f1 score achieved on the test set
    :param relabel_score: the score achieved by the relabeling function on the
                          test set
    :param task: the task the model was evaluated on
    """
    with open(options['log_file'], 'a') as f:
        print('Writing results to %s...' % options['log_file'])
        f.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%s\n' %
                (datetime.now().strftime(FORMAT), task, task_score, f1_score,
                 relabel_score, ' '.join(['%s=%s' % (opt, options[opt])
                                          for opt in options.keys()])))
