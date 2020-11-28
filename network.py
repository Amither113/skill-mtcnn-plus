#from flyai.train_helper import upload_data, download, sava_train_model  # 因为要蹭flyai的gpu
from dataset import AVAImages
from dataset_utils import dis2mean, get_index2score_dis, load_data
from network_utils import MTCNN_v3, MTCNN_v2, MTCNN, JSD, propagate_ROC, fixprob, tf_fixprob, read_cfg, get_W, ini_omega, \
    tr, r_kurtosis, style_loss, scalar_for_weights, update_omega, print_task_correlation, min_max_normalization, \
    get_cross_val_loss_transfer, get_all_train_accuracy, get_all_test_accuracy
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
slim = tf.contrib.slim


class Network(object):
    def __init__(self, input_size: tuple,
                 output_size: int):
        """
        init the network

        :param input_size: w, h, c
        :param output_size: size
        """
        self.input_size = input_size
        self.output_size = output_size

    # #######################################################TRAIN######################################################
    def train_pure_distribution(self, model_save_path='./model_score_CNN/', th_score=5.5):
        """
        训练单纯的分数分布预测模型
        :param data:
        :param model_save_path:
        :param val:
        :param task_marg:
        :param fix_marg:
        :return:
        """
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()
        dataset.load_dataset()
        dataset.val_set_y[:, 0: 10] = fixprob(dataset.val_set_y[:, 0: 10])

        # load parameters
        dataset.read_batch_cfg(task="TrainBatch")
        dataset.read_batch_cfg(task="TestBatch")
        dataset.read_batch_cfg(task="ThBatch")
        learning_rate, learning_rate_decay, epoch, alpha, gamma, theta = read_cfg(task="Skill-MTCNN")

        # placeholders
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
        y_outputs = MTCNN_v2(inputs=x, outputs=self.output_size, training=True)
        y_outputs_fix = tf_fixprob(y_outputs[:, 0: 10])

        # other parameters
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            r_kus = r_kurtosis(y_outputs_fix, th)
            dis_loss = JSD(y_outputs_fix, y[:, 0: 10])
            loss = r_kus * dis_loss
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                          var_list=train_theta + WW)

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        best_val_loss = 1000
        best_val_loss_step = 0
        stop_patience = 60
        best_loss = 1000
        best_loss_step = 0
        lr_patience = 50
        improvement_threshold = 0.999
        best_test_acc = 0.0
        best_test_acc_epoch = 0
        best_test_acc_batch = 0
        train_loss = []
        val_loss = []
        test_acc = []
        i = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while i <= epoch and not stop_flag:
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(flag="train")
                    y_b[:, 0: 10] = fixprob(y_b[:, 0: 10])
                    step = sess.run(global_step)

                    # cross_val_loss
                    th_end = 0
                    cross_val_loss_transfer = 0.0
                    while th_end == 0:
                        th_x_b, th_y_b, th_end = dataset.load_next_batch_quicker("Th")
                        th_y_b[:, 0: 10] = fixprob(th_y_b[:, 0: 10])
                        cross_val_loss_ = sess.run(dis_loss, feed_dict={x: th_x_b, y: th_y_b})
                        cross_val_loss_transfer += cross_val_loss_
                    cross_val_loss_transfer /= dataset.th_batch_index_max

                    # train
                    train_op_, loss_ = sess.run([train_op_all, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})
                    train_loss.append(loss_)
                    if loss_ < best_loss:
                        best_loss = loss_
                        best_loss_step = step

                    val_loss_ = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y, th: cross_val_loss_transfer})
                    val_loss.append(val_loss_)
                    print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                          format(dataset.batch_index_max, loss_, val_loss_, i + 1, dataset.batch_index))

                    if val_loss_ < best_val_loss * improvement_threshold:
                        best_val_loss = val_loss_
                        best_val_loss_step = step
                        # saver.save(sess, model_save_path + 'my_model')

                        # test acc
                        test_end = 0
                        test_correct_count = 0
                        while test_end == 0:
                            test_x_b, test_y_b, test_end = dataset.load_next_batch_quicker("test")
                            y_outputs_ = sess.run(y_outputs, feed_dict={x: test_x_b})
                            y_outputs_mean = dis2mean(y_outputs_[:, 0: 10])
                            y_pred = np.int64(y_outputs_mean >= th_score)

                            y_test = dis2mean(test_y_b[:, 0: 10])
                            y_test = np.int64(y_test >= th_score)  # 前提test_set_y.shape=(num,)

                            test_correct_count += sum((y_pred - y_test) == 0)
                        test_acc_ = test_correct_count / dataset.test_total
                        print("    test acc {acc} with best acc {best} in epoch{e}/batch{b}".
                              format(acc=test_acc_, best=best_test_acc, e=best_test_acc_epoch, b=best_test_acc_batch))

                        if test_acc_ > best_test_acc:
                            best_test_acc = test_acc
                            best_test_acc_epoch = i
                            best_test_acc_batch = dataset.batch_index

                    test_acc.append(test_acc_)

                    # 如果连着几个batch的train loss都没下降，调整学习率
                    if step - best_loss_step > lr_patience:
                        learning_rate *= 0.1

                    # 如果连着几个batch的val loss都没下降，则停止训练
                    if step - best_val_loss_step > stop_patience:
                        stop_flag = True
                        break

                    if end == 1:
                        break

            #### save
            save_curve = train_loss, val_loss, test_acc
            with open(model_save_path + 'curve.pkl', 'wb') as f:
                pickle.dump(save_curve, f)
            file_name = "mtcnnv2_pd"
            os.system('zip -r {f}.zip ./'.format(f=file_name) + model_save_path)
            sava_train_model(model_file="{f}.zip".format(f=file_name), dir_name="./file", overwrite=True)
            upload_data("{f}.zip".format(f=file_name), overwrite=True)

    def train_mtcnn(self, model_save_path='./model_mtcnn/'):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()
        dataset.load_dataset()
        dataset.val_set_y[:, 0: 10] = fixprob(dataset.val_set_y[:, 0: 10])
        # load parameters
        dataset.read_batch_cfg(task="TrainBatch")
        dataset.read_batch_cfg(task="TestBatch")
        dataset.read_batch_cfg(task="ThBatch")
        learning_rate, learning_rate_decay, epoch, alpha, gamma, theta = read_cfg(task="Skill-MTCNN")
        # placeholders
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
            task_id = tf.placeholder(tf.int32)
        y_outputs = MTCNN_v2(inputs=x, outputs=self.output_size, training=True)
        y_outputs_fix = tf_fixprob(y_outputs[:, 0: 10])

        # other parameters
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            W = get_W()
            ini_omega(self.output_size)
            omegaaa = tf.get_default_graph().get_tensor_by_name('Loss/Omega/omega:0')
            tr_W_omega_WT = tr(W, omegaaa)
            r_kus = r_kurtosis(y_outputs_fix, th)
            dis_loss = JSD(y_outputs_fix, y[:, 0: 10])
            loss = r_kus * (dis_loss +
                            gamma * style_loss(y_outputs[:, 10:], y[:, 10:]) +
                            tf.contrib.layers.apply_regularization(
                                regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
                                weights_list=tf.trainable_variables()) +
                            theta * tr_W_omega_WT)
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            opt = tf.train.AdamOptimizer(learning_rate)
            gradient_var_all = opt.compute_gradients(loss, var_list=train_theta + WW)
            capped_gvs = [(scalar_for_weights(grad, var, omegaaa, task_id), var)
                          for grad, var in gradient_var_all]
            train_op = opt.apply_gradients(capped_gvs)
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=train_theta + WW)
            train_op_omega = tf.assign(omegaaa, update_omega(W))

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        train_theta_and_W_first = 20
        best_loss = 1000
        best_loss_step = 0
        lr_patience = 50
        best_val_loss = 1000
        best_val_loss_step = 0
        stop_patience = 60
        stop_flag = False
        improvement_threshold = 0.999
        best_test_acc = 0.0
        best_test_acc_epoch = 0
        best_test_acc_batch = 0
        i = 0
        train_loss = []
        train_acc_all = []
        train_acc_batch = []
        val_loss = []
        test_acc = []
        test_acc_ = 0.0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while i <= epoch and not stop_flag:
                while True:
                    step = sess.run(global_step)

                    if step <= train_theta_and_W_first:
                        # 遍历所有batch
                        x_b, y_b, end = dataset.load_next_batch_quicker(flag="train")
                        y_b[:, 0: 10] = fixprob(y_b[:, 0: 10])

                        # train loss
                        cross_val_loss_transfer = get_cross_val_loss_transfer(sess, dataset, dis_loss, x, y)
                        train_op_, loss_ = sess.run([train_op_all, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})
                        train_loss.append(loss_)

                        # train accuracy batch
                        y_outputs_ = sess.run(y_outputs, feed_dict={x: x_b})
                        y_pred_ = np.argmax(y_outputs_[:, 0: 10], axis=1)
                        y_pred_ = np.int64(y_pred_ >= 5)
                        y_b_ = np.int64(np.argmax(y_b[:, 0: 10], axis=1) >= 5)
                        acc_batch_ = sum((y_pred_ - y_b_) == 0) / y_b.shape[0]
                        train_acc_batch.append(acc_batch_)

                        # train accuracy all
                        train_acc_all_ = get_all_train_accuracy(sess, y_outputs, x)
                        train_acc_all.append(train_acc_all_)

                        if loss_ < best_loss:
                            best_loss = loss_
                            best_loss_step = step

                    elif np.random.rand() < 0.5:
                        train_op_ = sess.run(train_op_omega)
                        sess.run(upgrade_global_step)
                        continue
                    else:
                        # 遍历所有batch
                        x_b, y_b, end = dataset.load_next_batch_quicker(flag="train")
                        y_b[:, 0: 10] = fixprob(y_b[:, 0: 10])

                        for taskid in range(self.output_size):
                            cross_val_loss_transfer = get_cross_val_loss_transfer(sess, dataset, dis_loss, x, y)
                            train_op_, loss_ = sess.run([train_op, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer, task_id: taskid})

                            if loss_ < best_loss:
                                best_loss = loss_
                                best_loss_step = step

                        train_loss.append(loss_)

                        # train accuracy batch
                        y_outputs_ = sess.run(y_outputs, feed_dict={x: x_b})
                        y_pred_ = np.argmax(y_outputs_[:, 0: 10], axis=1)
                        y_pred_ = np.int64(y_pred_ >= 5)
                        y_b_ = np.int64(np.argmax(y_b[:, 0: 10], axis=1) >= 5)
                        acc_batch_ = sum((y_pred_ - y_b_) == 0) / y_b.shape[0]
                        train_acc_batch.append(acc_batch_)

                        # train accuracy all
                        train_acc_all_ = get_all_train_accuracy(sess, y_outputs, x)
                        train_acc_all.append(train_acc_all_)

                    val_loss_ = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y, th: cross_val_loss_transfer})
                    val_loss.append(val_loss_)
                    print("epoch {ep} batch {b}/{bs} loss {loss}, validation loss {vl}".
                          format(ep=i+1, b=dataset.batch_index, bs=dataset.batch_index_max, loss=loss_, vl=val_loss_))

                    if val_loss_ < best_val_loss * improvement_threshold:
                        best_val_loss = val_loss_
                        best_val_loss_step = step  # 记录最小val所对应的batch index

                        # save data
                        # saver.save(sess, model_save_path + 'my_model')
                        # correlation matrix, cor1
                        Wa_and_Ws = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W'))
                        W = np.zeros(shape=(self.output_size, 4096))
                        for ii in range(W.shape[0]):
                            W[ii] = np.array(np.squeeze(Wa_and_Ws[ii * 2]))
                        cor_matrix1 = print_task_correlation(W, 10, self.output_size - 10)
                        cor_matrix1 = min_max_normalization(cor_matrix1)

                        # test acc
                        test_acc_ = get_all_test_accuracy(sess, y_outputs, dataset, x)
                        print("    test acc {acc} with best acc {best} in epoch{e}/batch{b}".
                              format(acc=test_acc_, best=best_test_acc, e=best_test_acc_epoch, b=best_test_acc_batch))

                        if test_acc_ >= best_test_acc:
                            best_test_acc = test_acc_
                            best_test_acc_epoch = i
                            best_test_acc_batch = dataset.batch_index

                    test_acc.append(test_acc_)

                    # 如果连着几个batch的train loss都没下降，调整学习率
                    if step - best_loss_step > lr_patience:
                        learning_rate *= 0.1

                    # 如果连着几个batch的val loss都没下降，则停止训练
                    if step - best_val_loss_step > stop_patience:
                        stop_flag = True
                        break

                    if end == 1:  # 一个epoch结束
                        break
                i += 1


    def train_cor_matrix_label(self, data='dataset/', model_save_path='./model_cor_matrix2/', val=True):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
            dataset.val_set_y[:, 10:] = self.fixprob(dataset.val_set_y[:, 10:])

        # load parameters
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = self.read_cfg()

        # placeholders
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, 10])
            y = tf.placeholder(tf.float32, [None, 14])

        # cor_fc_layer
        y_outputs = self.score2style(x)
        y_outputs = self.tf_fixprob(y_outputs)

        # other parameters
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            loss_c = self.JSD(y_outputs, y)
        with tf.name_scope("Train"):
            # get variables
            Wc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            train_op_wc = tf.train.AdamOptimizer(learning_rate).minimize(loss_c, global_step=global_step, var_list=Wc)

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # train wc
            print("training of Wc ...")
            best_val_loss = 1000
            improvement_threshold = 0.999
            patience = 4
            i = 0
            while i <= patience:
                while True:
                    _, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 10:] = self.fixprob(y_b[:, 10:])
                    sess.run(global_step)
                    if end == 1:
                        break

                    train_op_, loss_ = sess.run([train_op_wc, loss_c], feed_dict={x: y_b[:, 0: 10], y: y_b[:, 10:]})
                    if val:
                        val_loss = sess.run(loss_c, feed_dict={x: dataset.val_set_y[:, 0: 10], y: dataset.val_set_y[:, 10:]})
                        print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                              format(dataset.batch_index_max, loss_, val_loss, i + 1, dataset.batch_index))

                        if val_loss < best_val_loss * improvement_threshold:
                            patience *= 2
                            best_val_loss = val_loss
                i += 1

            # cor2
            cor_matrix2 = sess.run(tf.transpose(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')[0])
            )
            cor_matrix2 = self.min_max_normalization(cor_matrix2)

            # ### save
            cv2.imwrite(model_save_path + "cor_matrix2.png",
                        cv2.resize(cor_matrix2 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            saver.save(sess, model_save_path + 'my_model')

    def train_cor_matrix_predict(self, data='dataset/', model_read_path='./model_MTCNN_v2', model_save_path='./model_cor_matrix2/'):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()

        # load parameters
        dataset.read_batch_cfg(task="Cor_Matrix")
        learning_rate, learning_rate_decay, epoch= self.read_cfg(task="Cor_Matrix")

        # placeholders
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x_train = tf.placeholder(tf.float32, [None, 10])
            y_train = tf.placeholder(tf.float32, [None, 14])
            x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN_v2(x, True)  # y_outputs = (None, 24)
        y_outputs_mtcnn = tf.concat(y_list, axis=1)
        # y_outputs_to_one_ori = y_outputs[:, 0: 10] / tf.reduce_sum(y_outputs[:, 0: 10], keep_dims=True)
        # y_outputs_to_one = self.tf_fixprob(y_outputs_to_one_ori)

        # cor_fc_layer
        y_outputs = self.score2style(x_train)
        y_outputs = self.tf_fixprob(y_outputs)

        # other parameters
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            loss_c = self.JSD(y_outputs, y_train)
        with tf.name_scope("Train"):
            # get variables
            Wc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            train_op_wc = tf.train.AdamOptimizer(learning_rate).minimize(loss_c, global_step=global_step, var_list=Wc)

        variables_to_restore = slim.get_variables_to_restore(include=['Theta', 'W'])  # 单引号指只恢复一个层。双引号会恢复含该内容的所有层。
        re_saver = tf.train.Saver(variables_to_restore)  # 如果这里不指定特定的参数，sess会把目前graph中所有都恢复
        tf_vars = tf.trainable_variables(scope="Cor_Matrix")
        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2, var_list=tf_vars)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_read_path)
            if ckpt and ckpt.model_checkpoint_path:
                re_saver.restore(sess, ckpt.model_checkpoint_path)
            un_init = tf.variables_initializer(self.get_uninitialized_variables(sess))  # 获取没有初始化(通过已有model加载)的变量
            sess.run(un_init)  # 对没有初始化的变量进行初始化并训练.

            # train wc
            print("training of Wc ...")
            best_loss = 1000
            patience = 4
            i = 0
            while i <= patience or i <= epoch:
                while True:
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    sess.run(global_step)
                    if end == 1:
                        break

                    y_predict = sess.run(y_outputs_mtcnn, feed_dict={x: x_b})
                    y_predict[:, 10:] = self.fixprob(y_predict[:, 10:])
                    train_op_, loss_ = sess.run([train_op_wc, loss_c], feed_dict={x_train: y_predict[:, 0: 10],
                                                                                  y_train: y_predict[:, 10:]})
                    print("epoch {e} batch {b_index}/{b} loss {loss}".
                          format(e=i+1, b_index=dataset.batch_index, b=dataset.batch_index_max, loss=loss_))
                if loss_ <= best_loss:
                    best_loss = loss_
                    patience *= 2
                i += 1

            # cor2
            cor_matrix2 = sess.run(tf.transpose(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')[0])
            )
            cor_matrix2 = self.min_max_normalization(cor_matrix2)

            # ### save
            cv2.imwrite(model_save_path + "cor_matrix2.png",
                        cv2.resize(cor_matrix2 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            saver.save(sess, model_save_path + 'my_model')
            os.system('zip -r myfile.zip ./' + model_save_path)
            sava_train_model(model_file="myfile.zip", dir_name="./file", overwrite=True)
            upload_data("myfile.zip", overwrite=True)
