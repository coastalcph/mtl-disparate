import tensorflow as tf
import numpy as np

def bicond_reader(placeholders, target_sizes, vocab_size, label_vocab_size, **options):
    emb_dim = options["emb_dim"]
    lab_emb_dim = options["lab_emb_dim"]

    # [batch_size, max_seq1_length]
    seq1 = placeholders['seq1']

    # [batch_size, max_seq2_length]
    seq2 = placeholders['seq2']

    # [batch_size, labels_size]
    targets = tf.to_float(placeholders['targets'])

    label_vocab_inds = placeholders['label_vocab_inds']

    init = tf.contrib.layers.xavier_initializer(uniform=True)

    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("word_embeddings", [vocab_size, emb_dim], dtype=tf.float32, initializer=init)

    with tf.variable_scope("embedders") as varscope:
        seq1_embedded = tf.nn.embedding_lookup(embeddings, seq1)
        varscope.reuse_variables()
        seq2_embedded = tf.nn.embedding_lookup(embeddings, seq2)

    with tf.variable_scope("conditional_reader_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_states = reader(seq1_embedded, placeholders['seq1_lengths'], emb_dim,
                            scope=varscope1, **options)

    with tf.variable_scope("conditional_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        outputs, states = reader(seq2_embedded, placeholders['seq2_lengths'], emb_dim, seq1_states, scope=varscope2, **options)

    # shape output: [batch_size, 2*emb_dim]
    if options["main_num_layers"] == 1:
        # shape states: [2, 2]
        output = tf.concat([states[0][1], states[1][1]], 1)
    else:
        # shape states: [2, num_layers, 2]
        output = tf.concat([states[0][-1][1], states[1][-1][1]], 1)

    if lab_emb_dim != 0:
        with tf.variable_scope("label_embeddings"):
            label_embeddings = tf.get_variable("label_embeddings", [label_vocab_size, lab_emb_dim], dtype=tf.float32, initializer=init)

    with tf.variable_scope("bicond_preds"):
        # output of sequence encoders is projected into separate output layers, one for each task
        scores_dict, loss_dict, predict_dict = {}, {}, {}
        # iterate over the tasks
        for k in target_sizes.keys():
            # use task name as variable scope
            with tf.variable_scope(k) as varscope_task:
                if options["task_specific_layer_size"] > 0:
                    with tf.variable_scope(k + "_task_spec_layer") as task_spec_layer_scope:
                        output = tf.contrib.layers.fully_connected(output, options["task_specific_layer_size"],
                                                                   weights_initializer=init,
                                                                   activation_fn=tf.tanh, scope=task_spec_layer_scope)
                if lab_emb_dim != 0:

                    # placeholders['label_vocab_inds'] contain the index of the labels and 0 elsewhere, e.g. [0, 0, 0, 4, 5, 6, 0, 0, ...]
                    # shape: [batch_size, num_tasks*num_labels, label_embed_dim]
                    labels_embedded = tf.nn.embedding_lookup(label_embeddings, label_vocab_inds)

                    output_dim = emb_dim*2
                    if options["task_specific_layer_size"] > 0:
                        output_dim = options["task_specific_layer_size"]

                    output, labels_embedded = pad_output(output, labels_embedded, output_dim, lab_emb_dim)

                    # get predictions with dot product between output and embedded labels.
                    scores = dotprod_with_lab_embs(output, labels_embedded, label_vocab_inds)

                    # boolean_mask returns a 1-d tensor, so we need to reshape
                    # works for all models since we compute the target sizes for all models
                    scores = tf.reshape(scores, [-1, target_sizes[k]])
                    loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=targets)
                    predict = tf.nn.softmax(scores)

                else:
                    label_embeddings = None
                    if options["l1_rate_main"] != 1.0 or options["l2_rate_main"] != 1.0:
                        with tf.variable_scope(k + "_l1l2_reg") as l1l2scope:
                            l1_l2 = tf.contrib.layers.l1_l2_regularizer(scale_l1=options["l1_rate_main"], scale_l2=options["l2_rate_main"], scope=l1l2scope)
                            scores = tf.contrib.layers.fully_connected(output, target_sizes[k], weights_initializer=init,
                                        activation_fn=tf.tanh, scope=varscope_task, weights_regularizer=l1_l2)  # target_size
                    else:
                        scores = tf.contrib.layers.fully_connected(output, target_sizes[k], weights_initializer=init, activation_fn=tf.tanh, scope=varscope_task) # target_size
                    loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=targets)
                    predict = tf.nn.softmax(scores)

                scores_dict[k] = scores
                loss_dict[k] = loss
                predict_dict[k] = predict

    return scores_dict, loss_dict, predict_dict, label_embeddings


def relabel_model(placeholders, target_sizes, input_size_feats, input_size_preds, label_embeddings, **options):
    lab_emb_dim = options["lab_emb_dim"]

    soft_or_hard = options['ltn_pred_type']
    hidd_layer_size = options['lel_hid_size']

    # [batch_size, num_tasks - 2]
    ltn_preds = placeholders['preds_for_ltn']

    if options["feature_sets"] != "predsonly":
        # [batch_size, num_features]
        features = placeholders['features']

    # [batch_size, labels_size]
    targets = tf.to_float(placeholders['targets'])

    label_vocab_inds = placeholders['label_vocab_inds']

    # for returning main task predictions for relabelling with EM
    targets_main = tf.to_float(placeholders['targets_main'])
    label_vocab_inds_main = placeholders['label_vocab_inds_main']

    with tf.variable_scope("ltn_preds"):
        # output of sequence encoders is projected into separate output layers, one for each task
        init = tf.contrib.layers.xavier_initializer(uniform=True)
        scores_dict, loss_dict, predict_dict, predict_main_dict = {}, {}, {}, {}
        # iterate over the tasks
        for k in target_sizes.keys():
            # use task name as variable scope
            with tf.variable_scope(k) as varscope_task:
                if options["feature_sets"] != "predsonly":
                    # concatenate the predictions with the features
                    if soft_or_hard == 'hard':
                        emb_size = input_size_feats + input_size_preds
                    else:
                        emb_size = input_size_feats + input_size_preds[k]

                    output = tf.reshape(tf.concat([ltn_preds, features], 1), [-1, emb_size])
                else:
                    if soft_or_hard == 'hard':
                        emb_size = input_size_preds
                    else:
                        emb_size = input_size_preds[k]
                        output = tf.reshape(ltn_preds, [-1, emb_size])


                if options["l1_rate_ltn"] != 1.0 or options["l2_rate_ltn"] != 1.0:
                    l1_l2 = tf.contrib.layers.l1_l2_regularizer(scale_l1=options["l1_rate_ltn"], scale_l2=options["l2_rate_ltn"])

                output_dim = emb_size

                if hidd_layer_size != 0:
                    if options["l1_rate_ltn"] != 1.0 or options["l2_rate_ltn"] != 1.0:
                        with tf.variable_scope(k + "_relabel_hidd_layer") as task_spec_relabel_layer_scope:
                            output = tf.contrib.layers.fully_connected(output, hidd_layer_size, weights_initializer=init, weights_regularizer=l1_l2, scope=task_spec_relabel_layer_scope)
                    else:
                        with tf.variable_scope(k + "_relabel_hidd_layer") as task_spec_relabel_layer_scope:
                            output = tf.contrib.layers.fully_connected(output, hidd_layer_size, weights_initializer=init, scope=task_spec_relabel_layer_scope)

                    output_dim = hidd_layer_size

                predict_main = None

                if options["lab_embs_for_ltn"]:

                    # placeholders['label_vocab_inds'] contain the index of the labels and 0 elsewhere, e.g. [0, 0, 0, 4, 5, 6, 0, 0, ...]
                    # shape: [batch_size, num_tasks*num_labels, label_embed_dim]
                    labels_embedded = tf.nn.embedding_lookup(label_embeddings, label_vocab_inds)

                    output_padded, labels_embedded = pad_output(output, labels_embedded, output_dim, lab_emb_dim)

                    # get predictions with dot product between output and embedded labels.
                    scores = dotprod_with_lab_embs(output_padded, labels_embedded, label_vocab_inds)

                    # boolean_mask returns a 1-d tensor, so we need to reshape
                    scores = tf.reshape(scores, tf.shape(targets))

                    # then we also want to return predictions for the main task
                    if options["relabel_with_ltn"]:
                        labels_embedded_main = tf.nn.embedding_lookup(label_embeddings, label_vocab_inds_main)

                        output_padded_main, labels_embedded_main = pad_output(output, labels_embedded_main, output_dim, lab_emb_dim)

                        # get predictions with dot product between output and embedded main task labels.
                        scores_main = dotprod_with_lab_embs(output_padded_main, labels_embedded_main, label_vocab_inds_main)

                        # boolean_mask returns a 1-d tensor, so we need to reshape
                        scores_main = tf.reshape(scores_main, tf.shape(targets_main))

                        predict_main = tf.nn.softmax(scores_main)


                else:
                    if options["l1_rate_ltn"] != 1.0 or options["l2_rate_ltn"] != 1.0:
                        scores = tf.contrib.layers.fully_connected(output, target_sizes[k], weights_initializer=init,
                                   activation_fn=tf.tanh, scope=varscope_task, weights_regularizer=l1_l2)  # target_size

                    else:
                        scores = tf.contrib.layers.fully_connected(output, target_sizes[k], weights_initializer=init, activation_fn=tf.tanh, scope=varscope_task) # target_size
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=targets)
                predict = tf.nn.softmax(scores)

                scores_dict[k] = scores
                loss_dict[k] = loss
                predict_dict[k] = predict
                predict_main_dict[k] = predict_main

    return scores_dict, loss_dict, predict_dict, predict_main_dict


def dotprod_with_lab_embs(output, labels_embedded, label_vocab_inds):
    # dot product needs to happen with all the labels
    # shape output_expanded: [batch_size, 1, emb_dim*2]
    # shape labels_expanded: [batch_size, num_labels, emb_dim*2]
    # shape comb_repr: [batch_size, num_labels, emb_dim*2]
    output_expanded = tf.expand_dims(output, 1)
    comb_repr = tf.multiply(output_expanded, labels_embedded)

    # we remove the embedding dimension so we just have the scores
    #  shape: [batch_size, num_tasks*num_labels]
    reduced_output = tf.reduce_sum(comb_repr, 2)
    # now we want to mask it so only the labels for the task for which we have training data is taken into account for the loss
    # ... but this doesn't work yet
    #  a vector of zeros: [0, 0, 0, 0, 0, ...]
    zeroes = tf.zeros_like(label_vocab_inds)
    #  a vector indicating where label indices != 0
    #  [False, False, False, True, True, True, False, False, ...]
    mask = tf.not_equal(label_vocab_inds, zeroes)
    scores = tf.boolean_mask(reduced_output, mask)

    return scores


def pad_output(output, labels_embedded, output_dim, lab_emb_dim):
    if output_dim > lab_emb_dim:
        howmany = output_dim - lab_emb_dim
        labels_embedded = tf.pad(labels_embedded, [[0, 0], [0, 0], [0, howmany]], constant_values=0)
    elif lab_emb_dim > output_dim:
        howmany = lab_emb_dim - output_dim
        output = tf.pad(output, [[0, 0], [0, howmany]], constant_values=0)
    return output, labels_embedded


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None, **options):
    """Dynamic bi-LSTM reader; can be conditioned with initial state of other rnn.

    Args:
        inputs (tensor): The inputs into the bi-LSTM
        lengths (tensor): The lengths of the sequences
        output_size (int): Size of the LSTM state of the reader.
        context (tensor=None, tensor=None): Tuple of initial (forward, backward) states
                                  for the LSTM
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        Outputs (tensor): The outputs from the bi-LSTM.
        States (tensor): The cell states from the bi-LSTM.
    """

    skip_connections = options["skip_connections"]
    attention = options["attention"]
    num_layers = options["main_num_layers"]
    drop_keep_prob = options["dropout_rate"]

    with tf.variable_scope(scope or "reader") as varscope:
        if options["rnn_cell_type"] == "layer_norm":
            cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(output_size)
            cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(output_size)
        elif options["rnn_cell_type"] == "nas":
            cell_fw = tf.contrib.rnn.NASCell(output_size)
            cell_bw = tf.contrib.rnn.NASCell(output_size)
        elif options["rnn_cell_type"] == "phasedlstm":
            cell_fw = tf.contrib.rnn.PhasedLSTMCell(output_size)
            cell_bw = tf.contrib.rnn.PhasedLSTMCell(output_size)
        else: #LSTM cell
            cell_fw = tf.contrib.rnn.LSTMCell(output_size, initializer=tf.contrib.layers.xavier_initializer())
            cell_bw = tf.contrib.rnn.LSTMCell(output_size, initializer=tf.contrib.layers.xavier_initializer())
        if num_layers > 1:
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers)

        if drop_keep_prob != 1.0:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=drop_keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=drop_keep_prob)

        if skip_connections == True:
            cell_fw = tf.contrib.rnn.ResidualWrapper(cell_fw)
            cell_bw = tf.contrib.rnn.ResidualWrapper(cell_bw)

        if attention == True:
            cell_fw = tf.contrib.rnn.AttentionCellWrapper(cell_fw, attn_length=10)
            cell_bw = tf.contrib.rnn.AttentionCellWrapper(cell_bw, attn_length=10)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # ( (outputs_fw,outputs_bw) , (output_state_fw,output_state_bw) )
        # in case LSTMCell: output_state_fw = (c_fw,h_fw), and output_state_bw = (c_bw,h_bw)
        # each [batch_size x max_seq_length x output_size]
        return outputs, states
