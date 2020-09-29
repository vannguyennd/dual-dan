from tensorflow.contrib import rnn
from flip_gradient import flip_gradient
from units_all import *
import time
from sklearn import metrics as mt
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
            # define rnn cells with tensor_flow
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

            # get rnn cell output
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
            # calculate gradients
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


_, _, target_test, target_test_labels = read_data(time_steps, sen_len, False)

today = time.strftime('%Y%m%d')
hour = time.strftime('%h')


def train_and_evaluate():
    """helper to run the model with different training modes."""

    p_num_hidden = 256  # the size of the lstm hidden state corresponding to the best trained model

    saved_dir = "./dan_model/" + 'dan_peg_png/'  # the directory stores the best trained model obtained from running dan.py file
    result_file = open('./dan_model/' + 'dan_peg_png.txt', 'a+')

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

        target_test_values = get_data_values(target_test, num_input_vocabulary)
        gen_target_test_batch = batch_generator([target_test_values, target_test_labels], int(tbs))
        t_size = int(target_test.shape[0] // tbs)

        full_y_predict_train = np.array([])
        full_y_target_train = np.array([])

        for t_target in range(t_size):
            batch_x, batch_y = gen_target_test_batch.__next__()
            batch_y = convert_to_int(batch_y)

            full_y_target_train = np.append(full_y_target_train, batch_y)

            t_l_lb_prediction = sess.run(model.lb_prediction, feed_dict={model.X_source_target: batch_x,
                                                                         model.Y_source_target: batch_y,
                                                                         model.train: False})

            full_y_predict_train = np.append(full_y_predict_train, t_l_lb_prediction)

        trg_test_acc = mt.accuracy_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
        trg_test_pre = mt.precision_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
        trg_test_f1 = mt.f1_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
        trg_test_re = mt.recall_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
        trg_test_auc = mt.roc_auc_score(y_true=full_y_target_train, y_score=full_y_predict_train)

        tn, fp, fn, tp = mt.confusion_matrix(y_true=full_y_target_train,
                                             y_pred=full_y_predict_train).ravel()

        if (fp + tn) == 0:
            fpr = -1.0
        else:
            fpr = float(fp) / (fp + tn)

        if (tp + fn) == 0:
            fnr = -1.0
        else:
            fnr = float(fn) / (tp + fn)

        print('fpr: %.5f ; fnr: %.5f ; trg_test_acc: %.5f ; trg_test_pre: %.5f ; trg_test_f1: %.5f '
              '; trg_test_re: %.5f ; trg_test_auc: %.5f' % (fpr, fnr, trg_test_acc, trg_test_pre, trg_test_f1,
                                                            trg_test_re, trg_test_auc))

        result_file.write("fpr: %.5f; " % fpr)
        result_file.write("fnr: %.5f; " % fnr)
        result_file.write("trg_test_acc: %.5f; " % trg_test_acc)
        result_file.write("trg_test_pre: %.5f; " % trg_test_pre)
        result_file.write("trg_test_f1: %.5f; " % trg_test_f1)
        result_file.write("trg_test_re: %.5f; " % trg_test_re)
        result_file.write("trg_test_auc: %.5f\n" % trg_test_auc)


print('Domain adaptation testing')

train_and_evaluate()
