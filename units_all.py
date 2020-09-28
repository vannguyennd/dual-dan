import tensorflow as tf
import numpy as np
from random import shuffle
import random
from collections import Counter


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def read_data(max_input_sequence_len, size_m_ins, is_source):
    if is_source:
        data_non = './data_sets_numberized/peg_non.txt'
        data_vul = './data_sets_numberized/peg_vul.txt'
        tr_te_size = 0.8
    else:
        data_non = './data_sets_numberized/png_non.txt'
        data_vul = './data_sets_numberized/png_vul.txt'
        tr_te_size = 0.8

    inputs_non = []
    outputs_non = []
    with open(data_non, 'r') as file:
        recs = file.readlines()

        for idx, rec in enumerate(recs):

            inp = rec.split(' \n')[0].split('::: ')[1]
            inp = inp.split(' ')
            non_input = []

            for t in inp:
                non_input.append(int(t))

            non_input_len = len(non_input) // size_m_ins
            if non_input_len > max_input_sequence_len:
                non_input = np.array(non_input).reshape([-1, size_m_ins])[:max_input_sequence_len, :]
            else:
                # max_input_sequence_len is the length of each file, 1801 stands for pad word
                non_input += [1801] * ((max_input_sequence_len - non_input_len) * size_m_ins)
                non_input = np.array(non_input).reshape([-1, size_m_ins])

            inputs_non.append(non_input)
            outputs_non.append(0)

    inputs_non = np.stack(inputs_non)
    outputs_non = np.array(outputs_non)

    inputs_vul = []
    outputs_vul = []
    with open(data_vul, 'r') as file:
        recs = file.readlines()
        for idx, rec in enumerate(recs):
            inp = rec.split(' \n')[0].split('::: ')[1]
            inp = inp.split(' ')
            vul_input = []

            for t in inp:
                vul_input.append(int(t))

            vul_input_len = len(vul_input) // size_m_ins
            if vul_input_len > max_input_sequence_len:
                vul_input = np.array(vul_input).reshape([-1, size_m_ins])[:max_input_sequence_len, :]
            else:
                vul_input += [1801] * ((max_input_sequence_len - vul_input_len) * size_m_ins)
                vul_input = np.array(vul_input).reshape([-1, size_m_ins])
            inputs_vul.append(vul_input)

            outputs_vul.append(1)

    inputs_vul = np.stack(inputs_vul)
    outputs_vul = np.array(outputs_vul)

    size_non = int(tr_te_size * inputs_non.shape[0])
    size_vul = int(tr_te_size * inputs_vul.shape[0])

    random.seed(123)
    shuffle(inputs_non)
    random.seed(123)
    shuffle(inputs_vul)

    train_data_non = inputs_non[:size_non]
    train_data_vul = inputs_vul[:size_vul]
    train_label_non = outputs_non[:size_non]
    train_label_vul = outputs_vul[:size_vul]

    test_data_non = inputs_non[size_non:]
    test_data_vul = inputs_vul[size_vul:]
    test_label_non = outputs_non[size_non:]
    test_label_vul = outputs_vul[size_vul:]

    data_train = np.concatenate((train_data_non, train_data_vul), axis=0)
    label_train = np.concatenate((train_label_non, train_label_vul), axis=0)

    data_test = np.concatenate((test_data_non, test_data_vul), axis=0)
    label_test = np.concatenate((test_label_non, test_label_vul), axis=0)

    return data_train, label_train, data_test, label_test


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    np.random.seed(123)
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shu_fle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shu_fle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shu_fle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def get_data_values(p_data, num_input_vocabulary):
    sum_data = []

    for ib_data in range(p_data.shape[0]):
        sub_data = p_data[ib_data]
        batch_data = []
        for if_data in range(sub_data.shape[0]):
            data = np.zeros(num_input_vocabulary)
            data[sub_data[if_data]] = 1
            batch_data.append(data)

        sum_data.append(np.array(batch_data))

    return np.stack(sum_data)


def get_data_values_advance(p_data, num_input_vocabulary):
    sum_data = []

    for ib_data in range(p_data.shape[0]):
        sub_data = p_data[ib_data]
        batch_data = []
        for if_data in range(sub_data.shape[0]):
            data = np.zeros(num_input_vocabulary)
            oc_data = Counter(sub_data[if_data])
            for idx in range(data.shape[0]):
                if oc_data[idx] != 0:
                    data[idx] = oc_data[idx]
            batch_data.append(data)

        sum_data.append(np.array(batch_data))

    return np.stack(sum_data)


def get_batch(p_inputs, p_outputs, batch_size):
    data_size = p_inputs.shape[0]
    sample_values = np.random.choice(data_size, batch_size, replace=True)

    return p_inputs[sample_values], p_outputs[sample_values]


def compute_acc_pn(l_lb_prediction, source_test_labels, batch_size):
    s_p_acc = 0
    c_p_acc = 0
    s_n_acc = 0
    c_n_acc = 0

    for idx in range(batch_size):
        if source_test_labels[idx] == 0:
            s_p_acc += 1
            if l_lb_prediction[idx] == 0:
                c_p_acc += 1
        else:
            s_n_acc += 1
            if l_lb_prediction[idx] == 1:
                c_n_acc += 1

    return c_p_acc, s_p_acc, c_n_acc, s_n_acc


def convert_to_int(source_test_labels):
    result_values = []

    for i_values in source_test_labels:
        result_values.append(int(i_values))

    return np.array(result_values)


def compute_metric(l_lb_prediction, source_test_labels, batch_size):
    tp, fn, fp, tn = 0, 0, 0, 0

    for idx in range(batch_size):
        if source_test_labels[idx] == 0:
            if l_lb_prediction[idx] == 0:
                tn += 1
        elif source_test_labels[idx] == 1:
            if l_lb_prediction[idx] == 1:
                tp += 1
            else:
                fn += 1
        else:
            print('error!')

    for i_dx in range(batch_size):
        if l_lb_prediction[i_dx] == 1:
            if source_test_labels[i_dx] != 1:
                fp += 1

    return tp, fp, tn, fn
