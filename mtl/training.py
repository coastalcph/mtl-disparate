import tensorflow as tf
from collections import defaultdict
import numpy as np
from mtl.nn import bicond_reader, relabel_model
import os
from sklearn.metrics import classification_report
from preproc.log_utils import log_results, task2score, postproc_stance
from sklearn.metrics import f1_score
from copy import deepcopy
from constants import TASK_NAMES_SHORT
import copy
from preproc import batch

def alternate_epochs(target_sizes, max_iter, train_feed_dicts):
    """Return a batch generator that returns one epoch per batch and then
    switches tasks."""
    for task in target_sizes.keys():
        for j in range(0, max_iter):
            yield task, train_feed_dicts[task][j]


def alternate_batches(target_sizes, max_iter, train_feed_dicts):
    """Return a batch generator that returns one batch per task and then
    switches tasks."""
    for j in range(0, max_iter):
        for task in target_sizes.keys():
            yield task, train_feed_dicts[task][j]


def balanced_mtl_training_loop(placeholders, target_sizes, train_feed_dicts, dev_feed_dicts, max_iter,
                               min_op, logits_dict, loss_dict, preds_dict, sess, **options):
    # trains a MTL model, samples equal amounts of training data from each task and weighs the updates equally
    # early stopping based on main task dev set

    max_epochs = options["max_epochs"]
    main_task = options["main_task"]
    early_stopping = options["early_stopping"]
    batch_iter = alternate_batches if options["alternate_batches"] else alternate_epochs

    main_task_dev_acc = []
    stopping_criteron_reached = False

    for i in range(1, max_epochs + 1):
        task2loss_all, task2correct_all = defaultdict(list), defaultdict(float)
        task2total, task2correct_dev_all = defaultdict(float), defaultdict(float)
        task2total_dev = defaultdict(float)
        for task, batch in batch_iter(target_sizes, max_iter, train_feed_dicts):
            _, current_loss, p = sess.run([min_op[task], loss_dict[task], preds_dict[task]], feed_dict=batch)
            task2loss_all[task].extend(current_loss)
            hits = [pp for ii, pp in enumerate(p) if np.argmax(pp) == np.argmax(batch[placeholders["targets"]][ii])]
            task2correct_all[task] += len(hits)
            task2total[task] += len(batch[placeholders["targets"]])

        for task in target_sizes.keys():
            if options['dev_res_during_training']:
                p_inds_dev, g_inds_dev = [], []
                for j, batch_dev in enumerate(dev_feed_dicts[task]):
                    p_dev = sess.run(preds_dict[task], feed_dict=batch_dev)

                    # this is for super detailed results -- maybe we don't want to print this every epoch later on
                    if i % 1 == 0:
                        pred_inds = [np.argmax(pp_dev) for pp_dev in p_dev]
                        p_inds_dev.extend(pred_inds)
                        gold_inds = [np.argmax(batch_dev[placeholders["targets"]][i_d]) for i_d, targ in
                                     enumerate(batch_dev[placeholders["targets"]])]
                        g_inds_dev.extend(gold_inds)

                    hits = [pp for k, pp in enumerate(p_dev) if
                            np.argmax(pp) == np.argmax(batch_dev[placeholders["targets"]][k])]
                    task2correct_dev_all[task] += len(hits)
                    task2total_dev[task] += len(batch_dev[placeholders["targets"]])

            # Randomise batch IDs, so that selection of batch is random
            np.random.shuffle(train_feed_dicts[task])
            np.random.shuffle(dev_feed_dicts[task])
            acc = task2correct_all[task] / task2total[task]
            acc_dev = 0
            if options['dev_res_during_training']:
                acc_dev = task2correct_dev_all[task] / task2total_dev[task]
            if task != main_task:
                print('Epoch %d :' % i, "Task: " + task, "Loss: ", np.mean(task2loss_all[task]), "Acc: ", acc, "Acc Dev: ",
                      acc_dev)
            else:
                print('Epoch %d :' % i, "Task: " + task, "Loss: ", np.mean(task2loss_all[task]), "Acc: ", acc, "Acc Dev: ",
                      acc_dev, "Previous Acc Dev: ", main_task_dev_acc)

            # too much information
            #if options['dev_res_during_training'] and i % 1 == 0:
            #    print(classification_report(g_inds_dev, p_inds_dev, target_names=target_labels[task]))

            if task == main_task:
                if acc_dev >= early_stopping and len(main_task_dev_acc) >= 3 and acc_dev < main_task_dev_acc[-3]:
                    print("Dev accuracy is smaller than 4 epochs ago, early stopping criteron reached.")
                    stopping_criteron_reached = True
                    break
                main_task_dev_acc.append(acc_dev)
        if stopping_criteron_reached == True:
            break

    return logits_dict, loss_dict, preds_dict




def balanced_mtl_with_ltn_training_loop(placeholders, target_sizes, train_feed_dicts, dev_feed_dicts, max_iter, min_op, min_op_ltn,
                                        logits_dict, loss_dict, preds_dict, logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict, label_to_labelvocab, sess, **options):


    max_epochs_ltn = options["max_epochs_ltn"]
    main_task = options["main_task"]
    early_stopping = options["early_stopping"]
    hard_or_soft = options['ltn_pred_type']
    batch_iter = alternate_batches if options["alternate_batches"] else alternate_epochs

    main_task_dev_acc = []
    stopping_criteron_reached = False

    augment_data_from_epoch = max_epochs_ltn
    if options["relabel_with_ltn"]:
        # extend the training loop - for post-LTN data augmentation
        augment_data_from_epoch = max_epochs_ltn
        max_epochs_ltn = max_epochs_ltn + options["max_epochs_after_ltn"]

    for i in range(1, max_epochs_ltn + 1):
        task2loss_all, task2correct_all = defaultdict(list), defaultdict(float)
        task2total, task2correct_dev_all = defaultdict(float), defaultdict(float)
        task2total_dev = defaultdict(float)
        task2loss_all_ltn, task2correct_all_ltn = defaultdict(list), defaultdict(float)
        task2total_ltn, task2correct_dev_all_ltn = defaultdict(float), defaultdict(float)

        # we collect these only if we want to relabel
        task2preds = defaultdict(list)
        batches_to_relab = []  # collect those we've covered in this round
        batch2task = defaultdict()  # this one is for error analysis

        batch_id = 0
        for task, batch in batch_iter(target_sizes, max_iter, train_feed_dicts):

            # this is just the normal training step - we minimise the loss on the task's own training data here
            #batch = train_feed_dicts[task][j]
            _, current_loss, p = sess.run([min_op[task], loss_dict[task], preds_dict[task]], feed_dict=batch)
            task2loss_all[task].extend(current_loss)
            hits = [pp for ii, pp in enumerate(p) if np.argmax(pp) == np.argmax(batch[placeholders["targets"]][ii])]
            task2correct_all[task] += len(hits)
            task2total[task] += len(batch[placeholders["targets"]])

            # now we apply the models for all the other tasks to the batch and collect the predictions
            # this is used as input to the LTN model, to determine which of the instances in the batch can
            # also be used as training data for any of the other tasks
            batch = get_preds_for_ltn(sess, batch, placeholders, target_sizes, task, main_task, preds_dict,
                                      hard_or_soft, label_to_labelvocab, options["lab_emb_dim"], options["model_type"])

            if task != main_task:  # then we want to have a relabelling model training step
                _, current_loss_ltn, p_ltn = sess.run([min_op_ltn[task], loss_dict_ltn[task], preds_dict_ltn[task]], feed_dict=batch)
                task2loss_all_ltn[task].extend(current_loss_ltn)
                hits_ltn = [pp for ii, pp in enumerate(p_ltn) if np.argmax(pp) == np.argmax(batch[placeholders["targets"]][ii])]
                task2correct_all_ltn[task] += len(hits_ltn)
                task2total_ltn[task] += len(batch[placeholders["targets"]])

            if task == main_task and options["relabel_with_ltn"] and i >= augment_data_from_epoch:
                p_ltn = sess.run(predict_main_dict[task], feed_dict=batch)
                task2preds[task].append(p_ltn)
                batches_to_relab.append(batch)
                batch2task[batch_id] = task

            batch_id += 1

        if options["relabel_with_ltn"] and i >= augment_data_from_epoch:
            # we need to apply the relabelling function on the main task data first here, then pass the results on as train_feed_dicts

            # Before we reshuffle, see if we should augment the main task data
            train_data_additional = relabel_data_with_ltn_preds(batches_to_relab, task2preds, options["batch_size"])
            train_feed_dicts[main_task].append(train_data_additional)

        for task in target_sizes.keys():
            p_inds_dev, g_inds_dev, p_ids_ltn = [], [], []
            if options['dev_res_during_training']:
                for batch_dev in dev_feed_dicts[task]:

                    batch_dev = get_preds_for_ltn(sess, batch_dev, placeholders, target_sizes, task, main_task, preds_dict,
                                                  hard_or_soft, label_to_labelvocab, options["lab_emb_dim"], options["model_type"])

                    p_dev = sess.run(preds_dict[task], feed_dict=batch_dev)
                    hits = [pp for k, pp in enumerate(p_dev) if
                        np.argmax(pp) == np.argmax(batch_dev[placeholders["targets"]][k])]
                    task2correct_dev_all[task] += len(hits)
                    task2total_dev[task] += len(batch_dev[placeholders["targets"]])

                    # this is for super detailed results -- maybe we don't want to print this every epoch later on
                    if i % 1 == 0:
                        pred_inds = [np.argmax(pp_dev) for pp_dev in p_dev]
                        p_inds_dev.extend(pred_inds)
                        gold_inds = [np.argmax(batch_dev[placeholders["targets"]][i_d]) for i_d, targ in
                                 enumerate(batch_dev[placeholders["targets"]])]
                        g_inds_dev.extend(gold_inds)

                        p_dev_ltn = sess.run(preds_dict_ltn[task], feed_dict=batch_dev)
                        pred_inds_ltn = [np.argmax(pp_dev) for pp_dev in p_dev_ltn]
                        p_ids_ltn.extend(pred_inds_ltn)


            # Randomise batch IDs, so that selection of batch is random
            np.random.shuffle(train_feed_dicts[task])
            np.random.shuffle(dev_feed_dicts[task])

            if options['dev_res_during_training']:
                acc, acc_dev = task2correct_all[task] / task2total[task], task2correct_dev_all[task] / task2total_dev[task]
            else:
                acc = task2correct_all[task] / task2total[task]
                acc_dev = 0.0

            try:
                acc_ltn_train = task2correct_all_ltn[task] / task2total_ltn[task]
            except ZeroDivisionError:
                acc_ltn_train = 0

            if options["model_type"] == "label-transfer" or (options["model_type"] == 'semi-supervised' and task != main_task):
                print('Epoch %d :' % i, "Task: " + task, "Loss: ", np.mean(task2loss_all[task]), "Acc: ", acc, "Acc Dev: ",
                      acc_dev, "Acc LTN Train: ", acc_ltn_train)
            else:
                print('Epoch %d :' % i, "Task: " + task, "Loss: ", np.mean(task2loss_all[task]), "Acc: ", acc, "Acc Dev: ",
                      acc_dev, "Previous Acc Dev: ", main_task_dev_acc, "Acc LTN Train: ", acc_ltn_train)


            if task == main_task:
                if acc_dev >= early_stopping and len(main_task_dev_acc) >= 3 and acc_dev < main_task_dev_acc[-3]:
                    print("Dev accuracy is smaller than 4 epochs ago, early stopping criteron reached.")
                    stopping_criteron_reached = True
                    break
                main_task_dev_acc.append(acc_dev)
        if stopping_criteron_reached == True:
            break

    return logits_dict, loss_dict, preds_dict, logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict


def relabel_data_with_ltn_preds(all_batches, preds_ltn, batch_size):
    print("Executing the data augmentation function")

    # flatten batches for easier handling
    batches_flattenend = defaultdict(list)
    for key in all_batches[0].keys():
        values_flat = [all_batches[i][key] for i in range(0, len(all_batches))]
        values_flatter = [ii for i in values_flat for ii in i]
        batches_flattenend[key] = np.stack(values_flatter, axis=0)
        print("number instances")
        print(len(batches_flattenend[key]))

    relabelled_insts = defaultdict(list)
    num_sampled = 0

    # for each task, get the predictions
    for task, preds in preds_ltn.items():

        # flatten the preds batches
        preds = [pp for p in preds for pp in p]
        stacked = np.stack(preds, axis=1)

        # for each row (which are the probabilities for each label), sort the row by descending value and
        # store the index of the original array in the transformed array
        stacked_sorted = np.argsort(-stacked)

        label2inds = defaultdict(list)

        labelindex = 0
        for row in stacked_sorted:
            # the number of instances to sample are the top 10%, equally distributed across the labels
            num_inst_to_samp = int(len(row) * 0.1 * (1/len(stacked_sorted)))
            # determine how many batches this makes. We only want full batches so we might sometimes take slightly
            # less than the top 10%
            number_batches_to_samp = int(num_inst_to_samp/batch_size)
            num_inst_to_samp_final = number_batches_to_samp*batch_size
            label2inds[labelindex] = row[:num_inst_to_samp_final]

            num_sampled += num_inst_to_samp_final

            # store which batches are useful so we can iterate over these afterwards
            for instid in row[:num_inst_to_samp]:

                for key in all_batches[0].keys():
                    if key.name.startswith("label_vocab_inds:"):
                        relabelled_insts[key].append(batches_flattenend[key][instid])

                    elif key.name.startswith("targets:"):
                        # re-initialise the targets, then set the one for the predicted label to 1
                        targets_here = np.zeros([len(batches_flattenend[key][instid])], np.int32)
                        targets_here[labelindex] = 1
                        relabelled_insts[key].append(targets_here)

                    else:
                        relabelled_insts[key].append(batches_flattenend[key][instid])

            labelindex += 1

    # now all the relabelled data is in relabelled_insts and we need to change it to batch format again
    rebatched_instances = batch.batch_feed_dicts(relabelled_insts, batch_size, num_sampled)

    return rebatched_instances


def get_preds_for_ltn(sess, batch, placeholders, target_sizes, task, main_task, preds_dict, hard_or_soft, label_to_labelvocab, lab_emb_dim, model_type):
    # get predictions on dev data for EM
    p_task_for_ltn = []
    if lab_emb_dim > 0:
        # we don't want to modify the original batch
        batch_copy = dict.copy(batch) #copy.copy(batch)
    for taskjj in target_sizes.keys():
        if model_type == 'semi-supervised' and (taskjj == task or taskjj == main_task):
            continue
        elif model_type == 'label-transfer' and (taskjj == task):
            if taskjj != main_task:
                continue
        if lab_emb_dim > 0:
            label_vocab_inds = np.array([label_to_labelvocab[taskjj] for i in range(0, len(batch[placeholders["seq1"]]))], np.int64)
            batch_copy[placeholders["label_vocab_inds"]] = label_vocab_inds
            p_jj = sess.run([preds_dict[taskjj]], feed_dict=batch_copy)
        else:
            p_jj = sess.run([preds_dict[taskjj]], feed_dict=batch)
        if hard_or_soft == 'hard':
            pred_inds = [np.argmax(pp, 1) for pp in p_jj]
        else:
            pred_inds = p_jj
        if p_task_for_ltn == []:
            p_task_for_ltn = pred_inds
        else:
            p_task_for_ltn.extend(pred_inds)

    if model_type == 'label-transfer' or task != main_task:  # then we want to have an LTN model training step
        # enter current predictions in feed_dicts so main model predictions can be used by LTN model
        if hard_or_soft == 'hard':
            preds_for_ltn = np.stack(p_task_for_ltn, 1)
        else:
            preds_for_ltn = np.concatenate(p_task_for_ltn, 1)
        batch[placeholders["preds_for_ltn"]] = preds_for_ltn

    return batch


def train(placeholders, target_sizes, train_feed_dicts, dev_feed_dicts, vocab, label_vocab, input_size_preds, num_preds_ltn, label_to_labelvocab, sess=None, **options):

    max_num_batches = {}
    for task in target_sizes.keys():
        max_num_batches[task] = len(train_feed_dicts[task])
    if label_vocab == None:
        label_vocab_len = 0
    else:
        label_vocab_len = len(label_vocab)

    # create model
    logits_dict, loss_dict, preds_dict, label_embeddings = bicond_reader(placeholders, target_sizes, len(vocab), label_vocab_len, **options)  # those return dicts where the keys are the task names

    optim = tf.train.RMSPropOptimizer(learning_rate=options["learning_rate"])

    if options["model_type"] == "semi-supervised" or options["model_type"] == "label-transfer":
        # additional TF model needed for estimating relabelling function
        logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict = relabel_model(placeholders, target_sizes, input_size_preds, num_preds_ltn, label_embeddings, **options)  # those return dicts where the keys are the task names
        min_op_ltn = {}
        for task in target_sizes.keys():
            min_op_ltn[task] = optim.minimize(tf.reduce_mean(loss_dict_ltn[task]))

    min_op = {}
    for task in target_sizes.keys():
        min_op[task] = optim.minimize(tf.reduce_mean(loss_dict[task]))
    # The maximum number of iterations should be based on the number of batches in the smallest training set
    max_iter = min(max_num_batches.values())
    print("Max number batches for each task:", max_num_batches)
    print("Randomly sampling one from", str(max_iter), "batches for each task every training epoch")

    tf.global_variables_initializer().run(session=sess)

    if options["save_model"] == True:
        saver = tf.train.Saver(max_to_keep=100)

    if options["model_type"] == "hard-sharing":
        logits_dict, loss_dict, preds_dict = balanced_mtl_training_loop(placeholders, target_sizes, train_feed_dicts,
                                                                        dev_feed_dicts, max_iter,
                                                                        min_op, logits_dict, loss_dict, preds_dict, sess, **options)

        logits_dict_ltn, loss_dict_ltn, preds_dict_ltn = {}, {}, {}

    elif options["model_type"] == "semi-supervised" or options["model_type"] == "label-transfer":
        # load pre-trained mtl model
        print("Check if pre-trained MTL model exists...")
        save_path = get_save_path(create_path=False, **options)
        if not os.path.exists(save_path):
            print("Save path", save_path, "does not exist. Training MTL model first.")
            logits_dict, loss_dict, preds_dict = balanced_mtl_training_loop(placeholders, target_sizes, train_feed_dicts,
                                                                        dev_feed_dicts, max_iter,
                                                                        min_op, logits_dict, loss_dict, preds_dict, sess, **options)
        else:
            print("Model already exists. Restoring model.")
            saver = tf.train.Saver(max_to_keep=100)
            saver.restore(sess, save_path + "/model.ckpt")
            print("Model " + save_path + "/model.ckpt" + " restored.")

        print("\nStarting LTN training...")
        logits_dict, loss_dict, preds_dict, logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict = balanced_mtl_with_ltn_training_loop(placeholders, target_sizes, train_feed_dicts,
                                                                        dev_feed_dicts, max_iter, min_op, min_op_ltn, logits_dict, loss_dict, preds_dict,
                                                                        logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict, label_to_labelvocab, sess, **options)

    if options["save_model"] == True:
        savepath = get_save_path(create_path=True, **options)
        print("Saving model at location:", savepath)
        saver.save(sess, savepath + "/model.ckpt")

    return logits_dict, loss_dict, preds_dict, logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict


def restore_trained_model(placeholders, target_sizes, train_feed_dicts, vocab, label_vocab_len, label_to_labelvocab, input_size_preds, num_preds_ltn, sess=None, **options):

    max_num_batches = {}
    for task in target_sizes.keys():
        max_num_batches[task] = len(train_feed_dicts[task])

    # create model
    logits_dict, loss_dict, preds_dict, label_embeddings = bicond_reader(placeholders, target_sizes, len(vocab), label_vocab_len, **options)  # those return dicts where the keys are the task names

    logits_dict_ltn = loss_dict_ltn = preds_dict_ltn = None
    if options["model_type"] == "semi-supervised":
        logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict = relabel_model(placeholders, target_sizes, input_size_preds, num_preds_ltn, label_embeddings, **options)  # those return dicts where the keys are the task names

    tf.global_variables_initializer().run(session=sess)

    saver = tf.train.Saver(max_to_keep=100)
    save_path = get_save_path(create_path=False, **options)
    if not os.path.exists(save_path):
        print("Save path", save_path, "does not exist. Model cannot be loaded. Aborting.")
        return "", "", ""
    saver.restore(sess, save_path + "/model.ckpt")
    print("Model " + save_path + "/model.ckpt" + " restored.")

    return logits_dict, loss_dict, preds_dict, logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict


def get_save_path(create_path=True, **options):

    tasks = ",".join([TASK_NAMES_SHORT[tsk] for tsk in options["tasks"]])
    if type(options["feature_sets"]) == list:
        features = ",".join(options["feature_sets"])
    else:
        features = options["feature_sets"]
    debug = False
    if str(options["debug"]):
        debug = True
    lab_embs_for_ltn = False
    if str(options["lab_embs_for_ltn"]):
        lab_embs_for_ltn = True
    skip_connections = False
    if str(options["skip_connections"]):
        skip_connections = True
    attention = False
    if str(options["attention"]):
        attention = True
    alternate_batches = False
    if str(options["alternate_batches"]):
        alternate_batches = True
    ltn_pred_type, lel_hid_size, max_ltn = "", "", ""
    if options["model_type"] != "hard_sharing":
        ltn_pred_type = options['ltn_pred_type']
        lel_hid_size = str(options["lel_hid_size"])
        max_ltn = str(options["max_epochs_ltn"])

    save_model_dir = "_".join([options["model_type"], ltn_pred_type, options["main_task"], tasks, str(debug),
                               str(options["num_instances"]), str(options["emb_dim"]), lel_hid_size,
                               str(options["task_specific_layer_size"]), str(options["lab_emb_dim"]),
                               str(skip_connections), features, str(options["main_num_layers"]),
                               str(options["rnn_cell_type"]),
                               str(lab_embs_for_ltn), str(attention), str(alternate_batches),
                               str(options["batch_size"]), str(options["max_epochs"]), max_ltn,
                               str(options["early_stopping"]), str(options["learning_rate"]),
                               str(options["l1_rate_main"]), str(options["l2_rate_main"]),
                               str(options["l1_rate_ltn"]), str(options["l2_rate_ltn"]),
                               str(options["dropout_rate"]), str(options["exp_id"])])

    save_path = os.path.abspath(os.path.join("./save/", save_model_dir))
    if create_path == True and not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path