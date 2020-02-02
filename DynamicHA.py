import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt

class DHA():
    def __init__(self,DataFram,train_step=1000,x_newW=None,test_per=0.1,R_constant = 1.0,lr=0.8,P=1.0,P_R2Cm=1.0,P_class=0.0001):
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
        self.unique =  self.DataFram.loc[:,0].unique()
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

        sort_index = np.argsort(self.y_train)
        self.x_train = self.x_train[sort_index]
        self.y_train = self.y_train[sort_index]

        self.split_num = []
        for class_i in self.unique:
            self.split_num.append(len(self.y_train[self.y_train==class_i]))
    #测试数据集 在空间W中到球心的距离
    def Cm_test_distance(self,testdata,Cm,w,b,R):
        space_test = np.matmul(testdata,w) + b
        m,_ = np.shape(space_test)
        distance = np.zeros([m],dtype=np.float32)
        for i in range(m):
            distance[i] = np.linalg.norm(space_test[i,:]-Cm)/R
        return distance

    def Acc(self, Cmpre_list, target):
        Cmpre_np = np.array(Cmpre_list)
        Cmpre_index = Cmpre_np.argmin(axis=0).reshape([-1])
        y = self.unique[Cmpre_index]
        correct_predict = np.equal(y, target)
        accuracy = np.mean(correct_predict)
        return accuracy

    def plot_R(self):
        R_history = np.array(self.R_history)
        plt.figure()
        for i in range(len(self.split_num)):
            plt.plot(R_history[:,i])
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
        names = locals()
        with tf.name_scope('Space'):
            with tf.name_scope('InitVariable'):
                #动态变量名设置
                with tf.name_scope('V'):
                    V = tf.placeholder(tf.float32, shape=[None, self.x_w], name='Input')
                with tf.name_scope('TrainingStep'):
                    training_step = tf.Variable(0, name='TrainStep', trainable=False)
                    learning_rate = tf.Variable(self.lr, name='TrainStep', trainable=False)
                with tf.name_scope('weight'):
                    weight = tf.Variable(tf.random_normal([self.x_w, self.x_newW]), name='W', trainable=True)

                with tf.name_scope('bias'):
                    bias = tf.Variable(tf.zeros([self.x_newW]), name='B', trainable=True)

                with tf.name_scope('R'):
                    for i in range(len(self.split_num)):
                        names['R{}'.format(i)] = tf.Variable(initial_value=self.R_constant, dtype=tf.float32, name='R{}'.format(i), trainable=True)

            with tf.name_scope('layer'):
                U = tf.matmul(V, weight) + bias
                for i in range(len(self.split_num)):
                    names['U{}'.format(i)] = i
                # U1,U2,U3 = tf.split(U, self.split_num, 0)
                for index,i in  enumerate(tf.split(U, self.split_num, 0)):
                    names['U{}'.format(index)] = i

            with tf.name_scope('circle'):
                for i in range(len(self.split_num)):
                    names['Cm{}'.format(i)] = tf.reduce_mean(names['U{}'.format(i)],axis=0)

            with tf.name_scope('loss_pow'):
                split_list = np.arange(0,len(self.split_num))
                for i in range(len(self.split_num)):
                    with tf.name_scope('loss_pow{}'.format(i)):
                        split_list_ = np.delete(split_list,i)
                        if len(split_list_)==1:
                            U_ = names['U{}'.format(split_list_[0])]
                        else:
                            U_ = tf.concat([names['U{}'.format(j)] for j in split_list_], 0)

                        g1n1 = self.g1n_term(names['U{}'.format(i)],
                                             names['Cm{}'.format(i)],
                                             names['R{}'.format(i)])
                        g2n1 = self.g2n_term(U_,
                                             names['Cm{}'.format(i)],
                                             names['R{}'.format(i)])
                        Rn = tf.where(tf.greater(0.0, names['R{}'.format(i)]),
                                      names['R{}'.format(i)] ,
                                      0)

                        # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }

                        names['loss{}_pow'.format(i)] = tf.pow(g1n1, 2) + tf.pow(g2n1, 2) + tf.pow(Rn, 2)
                loss_pow = 0
                for i in range(len(self.split_num)):
                    loss_pow = loss_pow + names['loss{}_pow'.format(i)]

            with tf.name_scope('lossR2Cm'):
                combine = list(itertools.combinations(np.arange(0, len(self.split_num)).tolist(), 2))
                loss_R2Cm = 0
                for i in combine:
                    with tf.name_scope('lossR2Cm{}{}'.format(i[0],i[1])):
                        Cm_normal = (names['R{}'.format(i[0])] + names['R{}'.format(i[1])]) \
                                    - tf.linalg.norm(names['Cm{}'.format(i[0])] - names['Cm{}'.format(i[1])])
                        loss_R2Cm = loss_R2Cm + tf.where(tf.greater(Cm_normal, 0), Cm_normal, 0)

            with tf.name_scope('loss_class'):
                loss_class = 0
                for i in range(len(self.split_num)):
                    loss_class = loss_class + tf.linalg.norm(names['U{}'.format(i)] - names['Cm{}'.format(i)])

            with tf.name_scope('loss_all'):
                loss_all = self.P * (loss_pow) + self.P_R2Cm * loss_R2Cm + self.P_class * loss_class

            with tf.name_scope('Optimizer'):
                learning_rate = tf.train.exponential_decay(learning_rate, training_step, decay_steps=100,
                                                           decay_rate=0.8)
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)

            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init_op)
                # writer = tf.summary.FileWriter("demo_class", sess.graph)
                # writer.close()
                self.loss_list = []
                self.R_history = []

                print('Enter train the Space........')

                for j in range(self.train_step):

                    _ = sess.run(train_op, feed_dict={V: self.x_train})
                    loss = sess.run(loss_all, feed_dict={V: self.x_train})

                    self.loss_list.append(loss)
                    num = len(self.split_num)
                    R = sess.run([names['R{}'.format(i)] for i in range(num)])
                    self.R_history.append(R)

                print(self.loss_list)
                with open('loss_wine_.txt', 'w') as f:
                    for l in self.loss_list:
                        f.write(str(l) + ',')
                f.close()
                weight_ = sess.run(weight)
                bias_ = sess.run(bias)
                circle,R_list,Cmpre_test = [],[],[]
                for i in range(len(self.split_num)):
                    circle_ = sess.run(names['Cm{}'.format(i)], feed_dict={V: self.x_train})
                    R_ = sess.run(names['R{}'.format(i)])
                    circle.append(circle_)
                    R_list.append(R_)
                    Cmpre = self.Cm_test_distance(self.x_test, circle_, weight_, bias_, R_)
                    # gaus_Cmpre_one = Gaussian_PDF(weight_, bias_, datatrain[:Nm1, :], datatest)
                    Cmpre_test.append(Cmpre)
                acc = self.Acc(Cmpre_test, self.y_test)
                print(acc)


if __name__ == "__main__":
    import time
    DataPath = r'.\Wine.csv'
    DataSet = pd.read_csv(DataPath,header=None)

    DHA_classifier = DHA(DataSet,train_step=300)
    t1 = time.time()
    DHA_classifier.gen_model()
    t2 = time.time()
    print('图生成及训练时间：%.2f s',(t2 - t1))
    DHA_classifier.plot_R()
    plt.show()



