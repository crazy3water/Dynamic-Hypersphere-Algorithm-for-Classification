import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools

class DHA():
    def __init__(self,DataFram,train_step=100,x_newW=None,test_per=0.1,R_constant = 1,lr=0.001,P=0.001,P_R2Cm=0.001,P_class=0.0001):
        """
        :param DataFram:  数据集
        :param train_step:训练步数
        :param test_per:  测试数据集的抽样比例
        :param R_constant:初始化半径
        :param x_newW:    W的变换维度
        :param lr:        优化器学习率
        :param P:         每个球体的惩罚系数
        :param P_R2Cm:    球体之间的惩罚系数
        :param P_class:   加速收敛系数
        """
        self.DataFram = DataFram
        self.train_step = train_step
        self.x_newW = x_newW
        self.lr = lr
        self.P = P
        self.P_R2Cm = P_R2Cm
        self.P_class = P_class
        self.R_constant = R_constant
        print('-----------------------数据预处理----------------------')
        self.preprocess(test_per)
        print('-----------------------预处理完成----------------------')
    #数据预处理
    def preprocess(self,per):
        unique =  self.DataFram.loc[:,0].unique()
        target = self.DataFram.loc[:,0].values
        data = self.DataFram.loc[:,1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).values

        if self.x_newW == None:
            n, self.x_w = data.shape
            self.x_newW = self.x_w
        else:
            n, self.x_w = data.shape
        random_index = np.random.choice(range(100), size=1, replace=False)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size=per,
                                                                          random_state=int(random_index))

        self.split_num = []
        for class_i in unique:
            self.split_num.append(len([self.y_train==class_i]))

    def Acc_3(self, Cmpre_one, Cmpre_two, Cmpre_three, target):
        m, _ = np.shape(Cmpre_one)
        y_out = []
        y_ex = []
        with tf.name_scope('Acc'):
            y = np.zeros([m])
            for i in range(len(y)):
                if Cmpre_one[i] < 1 and Cmpre_two[i] > 1 and Cmpre_three[i] > 1:
                    y[i] = 1
                else:
                    if Cmpre_two[i] < 1 and Cmpre_one[i] > 1 and Cmpre_three[i] > 1:
                        y[i] = 2
                    else:
                        if Cmpre_three[i] < 1 and Cmpre_one[i] > 1 and Cmpre_two[i] > 1:
                            y[i] = 3
                        else:
                            if Cmpre_three[i] > 1 and Cmpre_one[i] > 1 and Cmpre_two[i] > 1:
                                y_out.append(i)
                            else:
                                y_ex.append(i)
                            if Cmpre_one[i] > Cmpre_two[i]:
                                if Cmpre_two[i] > Cmpre_three[i]:
                                    y[i] = 3
                                else:
                                    y[i] = 2
                            else:
                                if Cmpre_one[i] > Cmpre_three[i]:
                                    y[i] = 3
                                else:
                                    y[i] = 1
            correct_predict = np.equal(y, target)
            accuracy = np.mean(correct_predict)

        return accuracy

    #在圆心外惩罚
    def g1n_term(self, var, center, Rm):
        g1n = tf.linalg.norm(var - center, axis=1) - Rm
        g1n_max = tf.clip_by_value(g1n, 0, 1e10)
        penalty = tf.reduce_mean(g1n_max)  # if res>0,penalty = res else penalty = 0
        return penalty

    # 在圆心内惩罚
    def g2n_term(self, var, center, Rm):
        g2n = Rm - tf.linalg.norm(var - center, axis=1)
        g2n_max = tf.clip_by_value(g2n, 0, 1e10)
        penalty = tf.reduce_mean(g2n_max)  # if res>0,penalty = res else penalty = 0
        return penalty
    #产生模型
    def gen_model(self):
        circle = []
        r = []

        with tf.name_scope('Space'):
            with tf.name_scope('InitVariable'):
                #动态变量名设置
                with tf.name_scope('V'):
                    V = tf.placeholder(tf.float32, shape=[None, self.x_w], name='Input')
                with tf.name_scope('TrainingStep'):
                    training_step = tf.Variable(5, name='TrainStep', trainable=False)
                    learning_rate = tf.Variable(self.lr, name='TrainStep', trainable=False)
                with tf.name_scope('weight'):
                    weight = tf.Variable(tf.random_normal([self.x_w, self.x_newW]), name='W', trainable=True)

                with tf.name_scope('bias'):
                    bias = tf.Variable(tf.zeros([self.x_newW]), name='B', trainable=True)

                with tf.name_scope('R'):
                    for i in range(len(self.split_num)):
                        names['R{}'.format(i)] = tf.Variable(initial_value=self.R_constant, dtype=tf.float32, name='R1', trainable=True)

            with tf.name_scope('layer'):
                U = tf.matmul(V, weight) + bias
                U_ = tf.split(U, self.split_num, 0)
                for i in range(len(self.split_num)):
                    names['U{}'.format(i)] = U_[i]

            with tf.name_scope('circle'):
                for i in range(len(self.split_num)):
                    names['Cm{}'.format(i)] = tf.reduce_mean(names['U{}'.format(i)],axis=0)

            with tf.name_scope('loss'):
                split_list = np.arange(0,len(self.split_num))
                for i in range(len(self.split_num)):
                    split_list_ = np.delete(split_list,i)
                    if len(split_list_)==1:
                        U_ = names['U{}'.format(split_list_[0])]
                    else:
                        U_ = tf.concat([names['U{}'.format(j)] for j in split_list_], 0)

                    g1n1 = self.g1n_term(names['U{}'.format(i)],
                                         names['Cm{}'.format(i)],
                                         names['R{}'.format(i)])
                    g2n1 = self.g1n_term(U_,
                                         names['Cm{}'.format(i)],
                                         names['R{}'.format(i)])
                    Rn1 = tf.where(tf.greater(names['R{}'.format(i)], 0),names['R{}'.format(i)], 0)

                    # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }

                    names['loss{}_pow'.format(i)] = tf.pow(g1n1, 2) + tf.pow(g2n1, 2) + tf.pow(Rn1, 2)

                combine = list(itertools.combinations(np.arange(0, len(self.split_num)).tolist(), 2))
                Cm_penalty_ = 0
                for i in combine:
                    with tf.name_scope('lossR2Cm_R{}{}'.format(i[0],i[1])):
                        Cm12_normal = (names['R{}'.format(i[0])] + names['R{}'.format(i[1])]) \
                                               - tf.linalg.norm(names['Cm{}'.format(i[0])] - names['Cm{}'.format(i[0])])
                        Cm_penalty_ = Cm_penalty_ + tf.where(tf.greater(Cm12_normal, 0), Cm12_normal, 0)

            # loss_pow = tf.Variable(0, name='loss_pow', trainable=False)
            loss_pow = 0
            for i in range(len(self.split_num)):
                with tf.name_scope('loss_pow'):
                    loss_pow = loss_pow + names['loss{}_pow'.format(i)]

            loss_class = 0
            with tf.name_scope('loss_R2Cm'):
                loss_R2Cm = Cm_penalty_
                for i in range(len(self.split_num)):
                    loss_class = loss_class + tf.linalg.norm(names['U{}'.format(i)] - names['Cm{}'.format(i)])

            with tf.name_scope('loss_all'):
                loss_all = self.P * (loss_pow) + self.P_R2Cm * loss_R2Cm + self.P_class * loss_class

            with tf.name_scope('train'):
                learning_rate = tf.train.exponential_decay(learning_rate, training_step, decay_steps=100,
                                                           decay_rate=0.8)
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)

            init_op = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init_op)

                loss_list = []
                R1_list = []
                R2_list = []
                R3_list = []
                print('Enter train the Space........')

                for j in range(self.train_step):
                    _ = sess.run(train_op, feed_dict={V: self.x_train})
                    loss = sess.run(loss_all, feed_dict={V: self.x_train})
                    loss_list.append(loss)
                    num = len(self.split_num)
                    R_1, R_2, R_3 = sess.run([names['R{}'.format(i)] for i in range(num)])
                    R1_list.append(R_1)
                    R2_list.append(R_2)
                    R3_list.append(R_3)
                print(loss_list)
                with open('loss_wine_.txt', 'w') as f:
                    for l in loss_list:
                        f.write(str(l) + ',')
                f.close()
                weight_ = sess.run(weight)
                bias_ = sess.run(bias)
                Circle1 = sess.run(names['Cm{}'.format(1)], feed_dict={V: self.x_train})

if __name__ == "__main__":
    DataPath = r'.\Wine.csv'
    DataSet = pd.read_csv(DataPath,header=None)
    names = locals()
    DHA_classifier = DHA(DataSet,train_step=100)
    DHA_classifier.gen_model()



