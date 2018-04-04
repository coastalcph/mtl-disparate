from collections import defaultdict

from preproc.vocab import Vocab
from preproc.batch import get_feed_dicts
from preproc.map import numpify, tokenize, lower, deep_map, deep_seq_map, map_to_targets
from preproc.data_reader import task2data_reader
from features.features import create_features
import numpy as np


def load_data(placeholders, target_labels, target_sizes, **options):
    batch_size = options["batch_size"]
    prepared_data = defaultdict(dict)
    feature_data = {}
    if options['ltn_pred_type'] == 'hard':
        num_preds_ltn = len(options["tasks"]) - 2  # relevant for softmax / output layer
    else:
        num_preds_ltn = {}
        total_preds_ltn = 0

    if options["lab_emb_dim"] != 0:
        total_num_labels = 0
        for task in options["tasks"]:
            total_num_labels += target_sizes[task]

    vocab, label_vocab = None, None
    label_to_labelvocab = defaultdict(list)
    for task in options["tasks"]:
        read_data = task2data_reader(task)
        data_train, data_dev, data_test = read_data(
            debug=options["debug"], num_instances=options["num_instances"])
        target_labels[task] = data_train["labels"]
        if options['ltn_pred_type'] == 'soft':
            total_preds_ltn += len(data_train["labels"])

        # add data for creating data features
        feature_data[task] = data_train.get("seq2", []) +\
            data_dev.get("seq2", []) +\
            data_test.get("seq2", [])

        label_to_labelvocab_task = None
        if options["lab_emb_dim"] != 0:
            if label_vocab is None:
                label_vocab = Vocab() # unk is id 0
            label_to_labelvocab_i = [0] * total_num_labels
            for taskl in data_train["labels"]:
                labid = label_vocab(task + "_" + str(taskl))
                label_to_labelvocab_i[labid-1] = (labid) # -1 because the first one is UNK
            label_to_labelvocab_task = label_to_labelvocab_i
            label_to_labelvocab[task] = label_to_labelvocab_task


        prepared_data[task]["train"], vocab, label_vocab = prepare_data(placeholders,
                                                           data_train, vocab, label_vocab, label_to_labelvocab_task)
        prepared_data[task]["dev"], vocab, label_vocab = prepare_data(placeholders,
                                                         data_dev, vocab, label_vocab, label_to_labelvocab_task)
        prepared_data[task]["test"], vocab, label_vocab = prepare_data(placeholders,
                                                          data_test, vocab, label_vocab, label_to_labelvocab_task)


    vocab.freeze()  # this makes sure that nothing further is added to the vocab, otherwise deep_map will extend it
    if label_vocab is not None:
        label_vocab.freeze()

    if options['model_type'] != 'hard-sharing' and options["feature_sets"] != "predsonly":
        # create a mapping of tasks to an array for each training example
        print("Creating features")
        task2features = create_features(options["feature_sets"], feature_data,
                                    vocab, options["features_path"])

    if options['model_type'] == 'label-transfer':
        for task in options["tasks"]:
            num_preds_ltn[task] = total_preds_ltn
            if task != options['main_task']:
                num_preds_ltn[task] = total_preds_ltn - len(target_labels[task])

            if options["lab_emb_dim"] > 0:
                num_preds_ltn[task] = len(label_vocab.id2sym.keys()) -1
                if task != options['main_task']:
                    num_preds_ltn[task] = len(label_vocab.id2sym.keys()) - 1 - len(target_labels[task])

    elif options['em_pred_type'] == 'soft':
        for task in options["tasks"]:
            num_preds_ltn[task] = total_preds_ltn - len(target_labels[task]) - len(target_labels[options['main_task']])

            if options["lab_emb_dim"] > 0:
                num_preds_ltn[task] = len(label_vocab.id2sym.keys()) -1 - len(target_labels[task]) - len(target_labels[options['main_task']])

    #print(num_preds_ltn)

    train_feed_dicts, dev_feed_dicts, test_feed_dicts = {}, {}, {}
    for task in options["tasks"]:

        # padding to same length and converting lists to numpy arrays
        train_data = numpify(prepared_data[task]["train"], pad=0)
        dev_data = numpify(prepared_data[task]["dev"], pad=0)
        test_data = numpify(prepared_data[task]["test"], pad=0)

        if options['model_type'] != 'hard-sharing':
            if options["feature_sets"] != "predsonly":
                # add the data features to the data splits
                train_size, dev_size, test_size = train_data['seq1'].shape[0], \
                    dev_data['seq1'].shape[0], test_data['seq1'].shape[0]
                train_data['features'] = task2features[task][0:train_size]
                dev_data['features'] = task2features[task][
                               train_size:(train_size+dev_size)]
                test_data['features'] = task2features[task][-test_size:]

            if options['ltn_pred_type'] == 'soft':
                num_pr_ltn = num_preds_ltn[task]
            else:
                num_pr_ltn = num_preds_ltn

            train_data['preds_for_ltn'] = np.zeros([len(train_data["seq1"]), num_pr_ltn], np.float32)
            dev_data['preds_for_ltn'] = np.zeros([len(dev_data["seq1"]), num_pr_ltn], np.float32)
            test_data['preds_for_ltn'] = np.zeros([len(test_data["seq1"]), num_pr_ltn], np.float32)

            if options["lab_emb_dim"] > 0 and options["lab_embs_for_ltn"] and options["relabel_with_ltn"]:
                # this is just so that we can get main task predictions from models for any task more easily, using the label emb representation
                targets_main_len = target_sizes[options["main_task"]]
                train_data["targets_main"] = np.zeros([len(train_data["seq1"]), targets_main_len], np.int32)
                dev_data["targets_main"] = np.zeros([len(dev_data["seq1"]), targets_main_len], np.int32)
                test_data["targets_main"] = np.zeros([len(test_data["seq1"]), targets_main_len], np.int32)

                train_data["label_vocab_inds_main"] = [label_to_labelvocab[options["main_task"]] for inst in train_data["targets"]]
                dev_data["label_vocab_inds_main"] = [label_to_labelvocab[options["main_task"]] for inst in dev_data["targets"]]
                test_data["label_vocab_inds_main"] = [label_to_labelvocab[options["main_task"]] for inst in test_data["targets"]]

        train_feed_dicts[task] = get_feed_dicts(
            train_data, placeholders, batch_size=batch_size,
            inst_length=len(train_data["seq1"]))
        dev_feed_dicts[task] = get_feed_dicts(
            dev_data, placeholders, batch_size=batch_size,
            inst_length=len(dev_data["seq1"]))
        test_feed_dicts[task] = get_feed_dicts(
            test_data, placeholders, batch_size=batch_size,
            inst_length=len(test_data["seq1"]))

    return train_feed_dicts, dev_feed_dicts, test_feed_dicts, vocab, label_vocab, num_preds_ltn, label_to_labelvocab


def prepare_data(placeholders, data, vocab=None, label_vocab=None, label_to_labelvocab=None):
    data_tokenized = deep_map(data, tokenize, ['seq1', 'seq2'])
    data_lower = deep_seq_map(data_tokenized, lower, ['seq1', 'seq2'])
    data = deep_seq_map(data_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ["seq1", "seq2"])
    if vocab is None:
        vocab = Vocab()
        for instance in data["seq1"] + data["seq2"]:
            for token in instance:
                vocab(token)

    data = map_to_targets(data, "labels", "stance")  # map stance IDs to one-hot vectors, save in data["targets"]
    if label_vocab != None: # then we want label embeddings
        data["label_vocab_inds"] = [label_to_labelvocab for inst in data["targets"]]
    data_ids = deep_map(data, vocab, ["seq1", "seq2"])
    data_ids = deep_seq_map(data_ids, lambda xs: len(xs), keys=['seq1', 'seq2'], fun_name='lengths', expand=True)

    # removing data that's not a placeholder
    popl = []
    for k in data_ids.keys():
        if not k in placeholders.keys():
            popl.append(k)
    for p in popl:
        data_ids.pop(p, None)

    return data_ids, vocab, label_vocab
