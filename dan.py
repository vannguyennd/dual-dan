from tensorflow.contrib import rnn
from flip_gradient import flip_gradient
from units_all import *
import time
import os
import scipy.io as io

opt_type = 'adam'
cell_type = 'lstm'
num_layers = 1

tbs = 100
num_input_vocabulary = 4025
sen_len = 25
time_steps = 35
dense_output = 300
class_labels = 2
domain_labels = 2

cl_n_first = 300
cl_n_second = 300
d_n_first = 300
d_n_second = 300

num_steps = 150
steps = 10000
max_gradient_norm = 1.0


class DomainModel(object):
    """domain adaptation model."""

    def __init__(self, num_hidden):
        self._build_model(num_hidden)

    def _build_model(self, num_hidden):

        tf.reset_default_graph()

        self.X_source_target = tf.placeholder(tf.float32, shape=(None, time_steps, num_input_vocabulary))
        self.Y_source_target = tf.placeholder(tf.int32, shape=[None])
        self.Y_domain = tf.placeholder(tf.int32, shape=[None])

        self.ll = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        self.weights_st = weight_variable([2 * num_hidden * time_steps, dense_output])
        self.biases_st = bias_variable([dense_output])

        with tf.name_scope('source_target_network'):
            x_st = tf.unstack(self.X_source_target, time_steps, 1)
            # define rnn-cell with tensor_flow
            # forward direction cell
            if cell_type == 'lstm':
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(num_hidden) for _ in range(num_layers)])
                    # backward direction cell
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(num_hidden) for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.LSTMCell(
                        num_hidden)
                    # backward direction cell
                    bw_cell_st = rnn.LSTMCell(
                        num_hidden)
            elif cell_type == 'gru':
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(num_hidden) for _ in range(num_layers)])
                    # backward direction cell
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(num_hidden) for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.GRUCell(num_hidden)
                    # backward direction cell
                    bw_cell_st = rnn.GRUCell(num_hidden)
            else:
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(num_hidden) for _ in range(num_layers)])
                    # backward direction cell
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(num_hidden) for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.BasicRNNCell(num_hidden)
                    # backward direction cell
                    bw_cell_st = rnn.BasicRNNCell(num_hidden)

            # get rnn-cell outputs
            l_outputs_st, a_st, b_st = rnn.static_bidirectional_rnn(fw_cell_st, bw_cell_st, x_st, dtype=tf.float32,
                                                                    scope='fw_cell_st')

            l_outputs_st = tf.transpose(tf.stack(l_outputs_st, axis=0), perm=[1, 0, 2])
            l_outputs_st = tf.reshape(l_outputs_st, [-1, 2 * num_hidden * time_steps])

            outputs_st = tf.matmul(l_outputs_st, self.weights_st) + self.biases_st
            lo_gits_st = tf.reshape(outputs_st, [-1, dense_output])

            self.features = lo_gits_st

        with tf.name_scope('train_class_er'):

            all_features = lambda: self.features
            source_features = lambda: tf.slice(self.features, [0, 0], [int(tbs / 2), -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.Y_source_target
            source_labels = lambda: tf.slice(self.Y_source_target, [0], [int(tbs / 2)])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            w_cl_0 = weight_variable([dense_output, cl_n_first])
            b_cl_0 = bias_variable([cl_n_first])
            cl_h_fc0 = tf.nn.relu(tf.matmul(classify_feats, w_cl_0) + b_cl_0)

            w_cl_1 = weight_variable([cl_n_first, cl_n_second])
            b_cl_1 = bias_variable([cl_n_second])
            cl_h_fc1 = tf.nn.relu(tf.matmul(cl_h_fc0, w_cl_1) + b_cl_1)

            w_cl_2 = weight_variable([cl_n_second, 2])
            b_cl_2 = bias_variable([2])
            lo_gits_cl = tf.matmul(cl_h_fc1, w_cl_2) + b_cl_2

            self.prediction_l = tf.nn.softmax(lo_gits_cl)
            self.loss_op_l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lo_gits_cl,
                                                                            labels=self.classify_labels)

        with tf.name_scope('train_domain_er'):
            # flip the gradient when back-propagating through this operation
            feat = flip_gradient(self.features, self.ll)

            w_d_0 = weight_variable([dense_output, d_n_first])
            b_d_0 = bias_variable([d_n_first])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, w_d_0) + b_d_0)

            w_d_1 = weight_variable([d_n_first, d_n_second])
            b_d_1 = bias_variable([d_n_second])
            d_h_fc1 = tf.nn.relu(tf.matmul(d_h_fc0, w_d_1) + b_d_1)

            w_d_2 = weight_variable([d_n_second, 2])
            b_d_2 = bias_variable([2])
            lo_gits_d = tf.matmul(d_h_fc1, w_d_2) + b_d_2

            self.domain_prediction = tf.nn.softmax(lo_gits_d)
            self.loss_op_d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lo_gits_d, labels=self.Y_domain)

        with tf.name_scope("train_predict"):

            self.learning_rate = tf.placeholder(tf.float32, [])
            self.d_rate = tf.placeholder(tf.float32, [])

            prediction_loss = tf.reduce_mean(self.loss_op_l)
            domain_loss = tf.reduce_mean(self.loss_op_d)

            self.total_loss = prediction_loss + self.d_rate*domain_loss

            # get all trainable variables
            parameters = tf.trainable_variables()
            # Calculate gradients
            gradients = tf.gradients(self.total_loss, parameters)
            # clip gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

            if opt_type == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif opt_type == 'grad':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            self.dan_train_op = optimizer.apply_gradients(zip(clipped_gradients, parameters))

            correct_label_prediction = tf.nn.in_top_k(self.prediction_l, self.classify_labels, 1)
            self.label_acc = tf.reduce_mean(tf.cast(correct_label_prediction, tf.float32))

            correct_domain_prediction = tf.nn.in_top_k(self.domain_prediction, self.Y_domain, 1)
            self.domain_acc = tf.reduce_mean(tf.cast(correct_domain_prediction, tf.float32))

            self.lb_prediction = tf.argmax(self.prediction_l, 1)

            with tf.name_scope("init_save"):
                # initialize the variables (i.e. assign their default value)
                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=150)


source_train, source_train_labels, source_test, source_test_labels = read_data(time_steps, sen_len, True)
target_train, target_train_labels, target_test, target_test_labels = read_data(time_steps, sen_len, False)

today = time.strftime('%Y%m%d')
hour = time.strftime('%h')
high_values = []


def train_and_evaluate(training_mode, p_lr, p_d_rate, p_num_hidden, verbose=True):
    """helper to run the model with different training modes."""

    saved_dir = "./dan_save/" + 'model/' + str(p_lr) + '-' + str(p_d_rate) + '-' + str(p_num_hidden) + '/'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    result_file = open('./dan_save/' + 'model/' + str(today) + "_" + str(hour)
                       + '_domain_peg_png.txt', 'a+')

    model = DomainModel(p_num_hidden)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # check whether we have the model trained or not
        check_point = tf.train.get_checkpoint_state(saved_dir)
        if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
            print("load model parameters from %s" % check_point.model_checkpoint_path)
            model.saver.restore(sess, check_point.model_checkpoint_path)
        else:
            # if not, we start to initialize the model
            print("create the model with fresh parameters")
            sess.run(model.init)

        gen_source_batch = batch_generator([source_train, source_train_labels], int(tbs / 2))
        gen_target_batch = batch_generator([target_train, target_train_labels], int(tbs / 2))

        source_test_values = get_data_values(source_test, num_input_vocabulary)
        gen_source_test_batch = batch_generator([source_test_values, source_test_labels], int(tbs))
        s_size = int(source_test.shape[0] // tbs)

        target_test_values = get_data_values(target_test, num_input_vocabulary)
        gen_target_test_batch = batch_generator([target_test_values, target_test_labels], int(tbs))
        t_size = int(target_test.shape[0] // tbs)
        h_value = 0

        y_domain_labels = np.concatenate(([np.tile([0], [int(tbs / 2)]), np.tile([1], [int(tbs / 2)])]))

        print('p_lr: ' + str(p_lr) + '; ' + 'p_d_rate: ' + str(p_d_rate) + '; ' + 'p_num_hidden: ' +
              str(p_num_hidden))
        result_file.write('p_lr: ' + str(p_lr) + '; ' + 'p_d_rate: ' + str(p_d_rate) + '; ' + 'p_num_hidden: ' +
                          str(p_num_hidden) + '\n')

        # training loop
        for i_step in range(num_steps):
            p = float(i_step) / steps
            ll = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1

            l_lr = p_lr
            lr = l_lr / (1. + 10 * p) ** 0.75

            if training_mode == 'dan':

                x0, y0 = gen_source_batch.__next__()
                x1, y1 = gen_target_batch.__next__()

                x_0 = get_data_values(x0, num_input_vocabulary)
                x_1 = get_data_values(x1, num_input_vocabulary)

                x = np.concatenate((x_0, x_1), axis=0)
                y = np.concatenate((y0, y1), axis=0)

                _, batch_loss, d_acc, p_acc = \
                    sess.run([model.dan_train_op, model.total_loss, model.domain_acc, model.label_acc],
                             feed_dict={model.X_source_target: x, model.Y_source_target: y,
                                        model.Y_domain: y_domain_labels, model.train: True, model.ll: ll,
                                        model.learning_rate: lr, model.d_rate: p_d_rate})

                if verbose and i_step % 1 == 0:
                    print('epoch: ' + str(i_step))
                    result_file.write('epoch: ' + str(i_step) + '\n')
                    print('loss: %f  domain_acc: %f  prediction_acc: %f' % (batch_loss, d_acc, p_acc))
                    result_file.write('loss: %f  domain_acc: %f  prediction_acc: %f \n' % (batch_loss, d_acc, p_acc))

                    t_tp, t_fp, t_tn, t_fn = 0, 0, 0, 0
                    t_c_n_acc, t_s_n_acc, t_c_v_acc, t_s_v_acc = 0, 0, 0, 0

                    for t_target in range(t_size):
                        batch_x, batch_y = gen_target_test_batch.__next__()
                        batch_y = convert_to_int(batch_y)

                        t_l_lb_prediction = sess.run(model.lb_prediction, feed_dict={model.X_source_target: batch_x,
                                                                                     model.Y_source_target: batch_y,
                                                                                     model.train: False})

                        c_n_acc, s_n_acc, c_v_acc, s_v_acc = compute_acc_pn(t_l_lb_prediction, batch_y, tbs)
                        t_c_n_acc += c_n_acc
                        t_s_n_acc += s_n_acc
                        t_c_v_acc += c_v_acc
                        t_s_v_acc += s_v_acc

                        tp, fp, tn, fn = compute_metric(t_l_lb_prediction, batch_y, tbs)
                        t_tp += tp
                        t_fp += fp
                        t_tn += tn
                        t_fn += fn

                    if (t_fp + t_tn) == 0:
                        t_fpr = 0
                    else:
                        t_fpr = float(t_fp) / (t_fp + t_tn)

                    if (t_tp + t_fn) == 0:
                        t_fnr = 0
                    else:
                        t_fnr = float(t_fn) / (t_tp + t_fn)

                    if (t_tp + t_fn) == 0:
                        t_tpr = 0
                    else:
                        t_tpr = float(t_tp) / (t_tp + t_fn)

                    if (t_tp + t_fp) == 0:
                        t_p = 0
                    else:
                        t_p = float(t_tp) / (t_tp + t_fp)

                    if (t_p + t_tpr) == 0:
                        t_f_1 = 0
                    else:
                        t_f_1 = float(2 * t_p * t_tpr) / (t_p + t_tpr)


                    if t_f_1 > h_value:
                        h_value = t_f_1
                        if t_s_n_acc != 0 and t_s_v_acc != 0:
                            print('t_n_acc: %f; t_v_acc: %f' % (float(t_c_n_acc) / t_s_n_acc, float(t_c_v_acc) / t_s_v_acc))
                            result_file.write('t_n_acc: %f; t_v_acc: %f \n' % (float(t_c_n_acc) / t_s_n_acc,
                                                                           float(t_c_v_acc) / t_s_v_acc))
                        else:
                            print('accuracy divide by zero')
                        print('t_fpr: %f; t_fnr: %f; t_tpr: %f; t_p: %f; t_f_1: %f' % (t_fpr, t_fnr, t_tpr, t_p, t_f_1))
                        result_file.write('t_fpr: %f; t_fnr: %f; t_tpr: %f; t_p: %f; t_f_1: %f \n' % (t_fpr, t_fnr, t_tpr,
                                                                                                  t_p, t_f_1))

        if h_value != 0.0:
            high_values.append(h_value)
        print(high_values)

        result_file.write('\ntesting results: ')
        for i_value in high_values:
            result_file.write('%f \t' % i_value)
        result_file.write('\n')



print('Domain adaptation training')

g_lr = [0.001]
d_rate = [0.01, 0.1, 0.5, 1.0]
list_num_hidden = [128, 256]

for i_g_lr in g_lr:
    for i_d_rate in d_rate:
        for i_num_hidden in list_num_hidden:
            train_and_evaluate('dan', i_g_lr, i_d_rate, i_num_hidden)
