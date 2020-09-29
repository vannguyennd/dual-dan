from tensorflow.contrib import rnn
from flip_gradient import flip_gradient
from units_all import *
import time
from sklearn import metrics as mt
import os
import scipy.io as io

opt_type = 'adam'
cell_type = 'lstm'
num_layers = 1

tbs = 100
num_input_vocabulary = 4025
sen_len = 25
time_steps = 35
class_labels = 2
domain_labels = 1
hidden_dnn = 300
max_gradient_norm = 1.0
num_steps = 150

today = time.strftime('%Y%m%d')
hour = time.strftime('%h')


class DomainModel(object):
    """dual domain adaptation model."""

    def __init__(self, hidden_rnn):
        self._build_model(hidden_rnn)

    def classifier(self, x, reuse=None, scope=None):
        with tf.variable_scope(scope, reuse=reuse):
            w_dt_0 = weight_variable([2 * self.hidden_rnn, self.cl_n_first])
            b_dt_0 = bias_variable([self.cl_n_first])
            dt_h_fc0 = tf.nn.relu(tf.matmul(x, w_dt_0) + b_dt_0)

            w_dt_1 = weight_variable([self.cl_n_first, self.cl_n_second])
            b_dt_1 = bias_variable([self.cl_n_second])
            dt_h_fc1 = tf.nn.relu(tf.matmul(dt_h_fc0, w_dt_1) + b_dt_1)

            w_dt_2 = weight_variable([self.cl_n_second, 2])
            b_dt_2 = bias_variable([2])
            lo_gits_cl = tf.matmul(dt_h_fc1, w_dt_2) + b_dt_2

        return lo_gits_cl

    def generate_source_target(self, x, scope=None):
        with tf.variable_scope(scope):
            weights_st = weight_variable([2 * self.hidden_rnn * time_steps, 2 * self.hidden_rnn])
            biases_st = bias_variable([2 * self.hidden_rnn])
            if cell_type == 'lstm':
                if num_layers > 1:
                    # define rnn-cell with tensor_flow
                    # forward direction cell
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(self.hidden_rnn)
                                                              for _ in range(num_layers)])
                    # backward direction cell
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(self.hidden_rnn)
                                                              for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.LSTMCell(self.hidden_rnn)
                    # backward direction cell
                    bw_cell_st = rnn.LSTMCell(self.hidden_rnn)
            elif cell_type == 'gru':
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(self.hidden_rnn)
                                                              for _ in range(num_layers)])
                    # backward direction cell
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(self.hidden_rnn)
                                                              for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.GRUCell(self.hidden_rnn)
                    # backward direction cell
                    bw_cell_st = rnn.GRUCell(self.hidden_rnn)
            else:
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(self.hidden_rnn)
                                                              for _ in range(num_layers)])
                    # backward direction cell
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(self.hidden_rnn)
                                                              for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.BasicRNNCell(self.hidden_rnn)
                    # backward direction cell
                    bw_cell_st = rnn.BasicRNNCell(self.hidden_rnn)

            # get rnn-cell outputs
            l_outputs_st, a_st, b_st = rnn.static_bidirectional_rnn(fw_cell_st, bw_cell_st,
                                                                    x, dtype=tf.float32)
            l_outputs_st = tf.transpose(tf.stack(l_outputs_st, axis=0), perm=[1, 0, 2])

            l_outputs_st = tf.reshape(l_outputs_st, [-1, 2 * self.hidden_rnn * time_steps])

            outputs_st = tf.nn.tanh(tf.matmul(l_outputs_st, weights_st) + biases_st)
            lo_gits_st = tf.reshape(outputs_st, [-1, 2 * self.hidden_rnn])

            ab_st = tf.concat((a_st[1], b_st[1]), axis=1)

            return lo_gits_st, ab_st

    def discriminator_source_target(self, x, scope=None, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            w_dt_0 = weight_variable([2 * self.hidden_rnn, self.cl_n_first])
            b_dt_0 = bias_variable([self.cl_n_first])
            dt_h_fc0 = tf.nn.relu(tf.matmul(x, w_dt_0) + b_dt_0)

            w_dt_1 = weight_variable([self.cl_n_first, self.cl_n_second])
            b_dt_1 = bias_variable([self.cl_n_second])
            dt_h_fc1 = tf.nn.relu(tf.matmul(dt_h_fc0, w_dt_1) + b_dt_1)

            w_dt_2 = weight_variable([self.cl_n_second, 1])
            b_dt_2 = bias_variable([1])
            lo_gits_cl = tf.matmul(dt_h_fc1, w_dt_2) + b_dt_2

        return lo_gits_cl

    def _build_model(self, hidden_rnn):

        tf.reset_default_graph()

        self.X_st = tf.placeholder(tf.float32, shape=(tbs//2, time_steps, num_input_vocabulary))
        self.X_ts = tf.placeholder(tf.float32, shape=(tbs//2, time_steps, num_input_vocabulary))

        self.Y_st = tf.placeholder(tf.int32, shape=[tbs//2])
        self.Y_ts = tf.placeholder(tf.int32, shape=[tbs//2])

        self.Y_st_domain = tf.placeholder(tf.float32, shape=[tbs])
        self.Y_ts_domain = tf.placeholder(tf.float32, shape=[tbs])

        self.ld = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        self.sigma_kernel = tf.placeholder(tf.float32, [])

        self.cr_rate = tf.placeholder(tf.float32, [])
        self.d_rate = tf.placeholder(tf.float32, [])
        self.lp_rate = tf.placeholder(tf.float32, [])
        self.con_rate = tf.placeholder(tf.float32, [])
        self.mc_rate = tf.placeholder(tf.float32, [])

        self.cl_n_second = hidden_dnn
        self.cl_n_first = hidden_dnn
        self.dense_output = hidden_dnn
        self.hidden_rnn = hidden_rnn

        # global step
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope('generate_st'):
            x_s = tf.unstack(self.X_st, time_steps, 1)
            lo_gits_s, h_s = self.generate_source_target(x_s, scope='generate_st')

        with tf.name_scope('generate_ts'):
            x_t = tf.unstack(self.X_ts, time_steps, 1)
            lo_gits_t, h_t = self.generate_source_target(x_t, scope='generate_ts')

        self.features_st = lo_gits_s
        self.features_ts = lo_gits_t
        self.h_st = h_s
        self.h_ts = h_t

        with tf.name_scope('classifier_st'):
            lo_gits_st = self.classifier(self.features_st, scope='classifier')
            self.classify_labels_st = self.Y_st
            self.prediction_st = tf.nn.softmax(lo_gits_st)
            self.loss_op_l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lo_gits_st,
                                                                            labels=self.Y_st)

        with tf.name_scope('mode_collapse_source_target'):
            # for source domain data
            cross_s_sm_01 = tf.expand_dims(self.features_st, 1)
            cross_s_sm_02 = tf.expand_dims(cross_s_sm_01, 1)

            sum_kernel_s = cross_s_sm_01 - cross_s_sm_02
            sum_kernel_st = cross_s_sm_01 - cross_s_sm_02
            distance_source = tf.squeeze(tf.matmul(sum_kernel_s, sum_kernel_st, transpose_b=True))

            cross_hs_sm_01 = tf.expand_dims(self.h_st, 1)
            cross_hs_sm_02 = tf.expand_dims(cross_hs_sm_01, 1)

            sum_kernel_hs = cross_hs_sm_01 - cross_hs_sm_02
            sum_kernel_h_st = cross_hs_sm_01 - cross_hs_sm_02
            distance_h_source = tf.squeeze(tf.matmul(sum_kernel_hs, sum_kernel_h_st, transpose_b=True))

            niu_source = tf.exp((-self.sigma_kernel * distance_h_source))
            exp_x_source = tf.where(tf.is_nan(niu_source), tf.ones(niu_source.shape), niu_source)
            self.mc_source = exp_x_source * distance_source

            # for target domain data
            cross_t_sm_01 = tf.expand_dims(self.features_ts, 1)
            cross_t_sm_02 = tf.expand_dims(cross_t_sm_01, 1)

            sum_kernel_t = cross_t_sm_01 - cross_t_sm_02
            sum_kernel_tt = cross_t_sm_01 - cross_t_sm_02
            distance_target = tf.squeeze(tf.matmul(sum_kernel_t, sum_kernel_tt, transpose_b=True))

            cross_ht_sm_01 = tf.expand_dims(self.h_ts, 1)
            cross_ht_sm_02 = tf.expand_dims(cross_ht_sm_01, 1)

            sum_kernel_ht = cross_ht_sm_01 - cross_ht_sm_02
            sum_kernel_h_tt = cross_ht_sm_01 - cross_ht_sm_02
            distance_h_target = tf.squeeze(tf.matmul(sum_kernel_ht, sum_kernel_h_tt, transpose_b=True))

            niu_target = tf.exp((-self.sigma_kernel * distance_h_target))
            exp_x_target = tf.where(tf.is_nan(niu_target), tf.ones(niu_target.shape), niu_target)

            self.mc_target = exp_x_target * distance_target

        with tf.name_scope('train_domain_source_target'):
            # flip the gradient when back-propagating through this operation
            self.feat_source = flip_gradient(self.features_st, self.ld)
            lo_gits_d_source = self.discriminator_source_target(self.feat_source, scope='train_d_st')
            self.domain_st_sm = tf.nn.sigmoid(lo_gits_d_source)

            # flip the gradient when back-propagating through this operation
            self.feat_target = flip_gradient(self.features_ts, self.ld)
            lo_gits_d_target = self.discriminator_source_target(self.feat_target, scope='train_d_ts')
            self.domain_ts_sm = tf.nn.sigmoid(lo_gits_d_target)

            lo_gits_st = tf.concat((lo_gits_d_source, lo_gits_d_target), axis=0)
            self.loss_d_source = tf.nn.sigmoid_cross_entropy_with_logits(logits=lo_gits_st,
                                                                         labels=tf.reshape(self.Y_st_domain,
                                                                                           [-1, 1]))
            self.loss_d_target = tf.nn.sigmoid_cross_entropy_with_logits(logits=lo_gits_st,
                                                                         labels=tf.reshape(self.Y_ts_domain,
                                                                                           [-1, 1]))

        with tf.name_scope('train_con_source_target'):
            lo_gits_s = self.discriminator_source_target(self.feat_source, scope='train_d_ts',
                                                         reuse=True)
            lo_gits_t = self.discriminator_source_target(self.feat_target, scope='train_d_st',
                                                         reuse=True)

            self.con_t = -tf.log(tf.nn.sigmoid(lo_gits_s))
            self.con_s = -tf.log(tf.nn.sigmoid(lo_gits_t))

        with tf.name_scope('train_predict'):
            self.learning_rate = tf.placeholder(tf.float32, [])
            self.d_rate = tf.placeholder(tf.float32, [])

            self.prediction_loss = tf.reduce_mean(self.loss_op_l)

            self.domain_src_loss = tf.reduce_mean(self.loss_d_source)
            self.domain_trg_loss = tf.reduce_mean(self.loss_d_target)

            self.mc_src_loss = tf.reduce_mean(self.mc_source)
            self.mc_trg_loss = tf.reduce_mean(self.mc_target)

            self.con_src_loss = tf.reduce_mean(self.con_s)
            self.con_trg_loss = tf.reduce_mean(self.con_t)

            self.total_loss = \
                self.prediction_loss + self.d_rate * (self.domain_src_loss + self.domain_trg_loss) \
                + self.mc_rate * (self.mc_src_loss + self.mc_trg_loss) \
                + self.con_rate * (self.con_src_loss + self.con_trg_loss)

            # get all trainable variables
            parameters = tf.trainable_variables()
            # calculate gradients
            gradients = tf.gradients(self.total_loss, parameters)  # compute gradient on all parameters
            # clip gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

            if opt_type == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif opt_type == 'grad':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            self.dan_train_op = optimizer.apply_gradients(zip(clipped_gradients, parameters))

            correct_label_prediction = tf.nn.in_top_k(self.prediction_st, self.classify_labels_st, 1)
            self.label_acc = tf.reduce_mean(tf.cast(correct_label_prediction, tf.float32))
            tf.summary.scalar('classifier_acc: ', self.label_acc)

            correct_domain_st_sm = tf.equal(tf.round(self.domain_st_sm), self.Y_st_domain)
            self.domain_st_acc = tf.reduce_mean(tf.cast(correct_domain_st_sm, tf.float32))
            tf.summary.scalar('domain_st_acc: ', self.domain_st_acc)

            correct_domain_ts_sm = tf.equal(tf.round(self.domain_ts_sm), self.Y_ts_domain)
            self.domain_ts_acc = tf.reduce_mean(tf.cast(correct_domain_ts_sm, tf.float32))
            tf.summary.scalar('domain_ts_acc: ', self.domain_ts_acc)

            self.lb_prediction_st = tf.argmax(self.prediction_st, 1)

        with tf.name_scope("init_save"):
            # initialize the variables (i.e. assign their default value)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=150)


_, _, target_test, target_test_labels = read_data(time_steps, sen_len, False)


def train_and_evaluate():
    """helper to run the model with different training modes."""
    hidden_rnn = 128  # the size of the lstm hidden state corresponding to the best trained model
    saved_dir = "./dual_dan_model/" + 'dual_dan_peg_png_hg/'  # the directory stores the best trained model obtained from running dual_dan_HG.py file.

    model = DomainModel(hidden_rnn)

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
        gen_target_test_batch = batch_generator([target_test_values, target_test_labels], int(tbs // 2))
        t_size = int(target_test.shape[0] // (tbs // 2))

        result_file = open('./dual_dan_model/' + 'dual_dan_peg_png_hg.txt', 'a+')

        # for target
        full_y_predict_train = np.array([])
        full_y_target_train = np.array([])

        for t_target in range(t_size):
            batch_x, batch_y = gen_target_test_batch.__next__()
            batch_y = convert_to_int(batch_y)

            full_y_target_train = np.append(full_y_target_train, batch_y)

            t_l_lb_prediction = sess.run(model.lb_prediction_st, feed_dict={model.X_st: batch_x,
                                                                            model.Y_st: batch_y})

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
          '; trg_test_re: %.5f ; trg_test_auc: %.5f' % (fpr, fnr, trg_test_acc, trg_test_pre,
                                                        trg_test_f1, trg_test_re, trg_test_auc))
        result_file.write('fpr: %.5f ; fnr: %.5f ; trg_test_acc: %.5f ; trg_test_pre: %.5f ; trg_test_f1: %.5f '
          '; trg_test_re: %.5f ; trg_test_auc: %.5f \n' % (fpr, fnr, trg_test_acc, trg_test_pre,
                                                        trg_test_f1, trg_test_re, trg_test_auc))


print('dual domain adaptation testing')
train_and_evaluate()
