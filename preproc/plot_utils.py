import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np

from constants import FNC, STANCE, NLI, TOPIC, LAPTOP, RESTAURANT, TARGET,\
    TOPIC_5WAY, STANCE_LABELS, FNC_LABELS, NLI_LABELS, TOPIC_LABELS, \
    TOPIC_5WAY_LABELS, ABSA_LABELS, TARGET_LABELS


def task2labels(task):
    if task == STANCE:
        return STANCE_LABELS
    if task == FNC:
        return FNC_LABELS
    if task == NLI:
        return NLI_LABELS
    if task == TOPIC:
        return TOPIC_LABELS
    if task == TOPIC_5WAY:
        return TOPIC_5WAY_LABELS
    if task in [LAPTOP, RESTAURANT]:
        return ABSA_LABELS
    if task == TARGET:
        return TARGET_LABELS
    raise ValueError('No labels available for task %s.' % task)


def task2display_name(task):
    if task == STANCE:
        return 'Stance'
    if task == FNC:
        return 'FNC-1'
    if task == NLI:
        return 'MultiNLI'
    if task == TOPIC:
        return 'Topic-2'
    if task == TOPIC_5WAY:
        return 'Topic-5'
    if task == LAPTOP:
        return 'ABSA-L'
    if task == RESTAURANT:
        return 'ABSA-R'
    if task == TARGET:
        return 'Target'
    raise ValueError('%s is not a valid task.' % task)


def task2color(task):
    if task == TOPIC:
        return 'forestgreen'
    if task == TOPIC_5WAY:
        return 'yellowgreen'
    if task == LAPTOP:
        return 'cornflowerblue'
    if task == RESTAURANT:
        return 'mediumblue'
    if task == STANCE:
        return 'midnightblue'
    if task == TARGET:
        return 'saddlebrown'
    if task == FNC:
        return 'darkgoldenrod'
    if task == NLI:
        return 'slategray'
    raise ValueError('%s is not available.' % task)


def label2display_name(label):
    if label in ['AGAINST', 'FAVOR', 'NONE']:
        return label.lower()
    try:
        label = float(label)
        if label == 0:
            return 'neutral'
        if label == -1:
            return 'negative'
        if label == -2:
            return 'highly negative'
        if label == 1:
            return 'positive'
        if label == 2:
            return 'highly positive'
    except:
        return label
    return label


def plot_label_embeddings(sess, tasks, label_vocab):
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "label_embeddings/label_embeddings")
    assert len(var_list) > 0, 'Error: Label embeddings have not been saved.'
    assert len(var_list) == 1

    label_embeddings = sess.run(var_list[0])
    print('Loaded label embeddings of shape:', label_embeddings.shape)

    assert label_vocab is not None

    # remove the UNK label of the label embeddings
    label_embeddings = label_embeddings[1:, :]

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'olive', 'cyan', 'brown']

    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    pca = PCA(n_components=2)

    label_embeddings_tsne = pca.fit_transform(label_embeddings)
    label_names = []
    task_names = []
    for i, task in enumerate(tasks):
        task_labels = task2labels(task)
        label_names += task_labels
        task_names += [task] * len(task_labels)
    # as a sanity check, make sure that the labels correspond with those in the
    # label vocab; +1 because the labels start at 1 (0 is UNK)
    for i in range(label_embeddings.shape[0]):
        label_id = "%s_%s" % (task_names[i], str(label_names[i]))
        # print(i+1, label_id, label_vocab.sym2id[label_id])
        assert i+1 == label_vocab.sym2id[label_id],\
            'Error: Id %d != label id %d for %s.' % (i+1, label_id, task_names[i])

    file_name = 'label_embeddings.png'
    plot_embedding(label_embeddings_tsne, label_names, task_names, file_name=file_name)


def plot_embedding(X, y, tasks, title=None, file_name=None):
    """Plot an embedding X with the label y colored by colors."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # we can increase the resolution by increasing the figure size
    plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if tasks[i] == STANCE:
            # skip stance and plot later
            continue
        plt.text(X[i, 0], X[i, 1], label2display_name(str(y[i])),
                 color=task2color(tasks[i]),
                 fontdict={'weight': 'bold', 'size': 9})

    for i in range(X.shape[0]):
        if tasks[i] == STANCE:
            plt.text(X[i, 0], X[i, 1], label2display_name(str(y[i])),
                     color=task2color(tasks[i]),
                     fontdict={'weight': 'bold', 'size': 9})

    # create patches for the legend
    patches = []
    for task in sorted(list(set(tasks))):
        patches.append(mpatches.Patch(color=task2color(task), label=task2display_name(task)))
    lgd = plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1),
                     edgecolor='black')

    # plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    # plt.show()
    plt.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
